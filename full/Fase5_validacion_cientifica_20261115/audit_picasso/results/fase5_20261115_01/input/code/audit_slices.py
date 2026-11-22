#!/usr/bin/env python3
"""Independent first-exit tracing on exported Gaussian/material slices.

Requires NumPy only. This audits exported slices; it does not evolve Einstein,
certify a global Morse theorem, or validate the export's change of slicing.
Near-critical or unresolved cases never receive an invented outgoing endpoint.
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np


TWOPI = 2.0*np.pi


def wrap_error(a, b):
    return float(abs((a-b+np.pi) % TWOPI-np.pi))


def weights(u):
    w = np.array([-u*(u-1)*(u-2)/6, (u+1)*(u-1)*(u-2)/2,
                  -(u+1)*u*(u-2)/2, (u+1)*u*(u-1)/6])
    dw = np.array([-u*u/2+u-1/3, 3*u*u/2-2*u-1/2,
                   -3*u*u/2+u+1, u*u/2-1/6])
    ddw = np.array([1-u, 3*u-2, 1-3*u, u])
    return w, dw, ddw


class CubicGrid:
    def __init__(self, x, y, values):
        self.x, self.y, self.values = map(np.asarray, (x, y, values))
        assert self.values.shape == (len(x), len(y))
        assert min(len(x), len(y)) >= 4
        self.dx, self.dy = float(x[1]-x[0]), float(y[1]-y[0])
        assert self.dx > 0 and self.dy > 0
        assert np.allclose(np.diff(x), self.dx, rtol=1e-10, atol=1e-13)
        assert np.allclose(np.diff(y), self.dy, rtol=1e-10, atol=1e-13)

    def jet(self, point):
        x, y = point
        if not (self.x[0] <= x <= self.x[-1] and self.y[0] <= y <= self.y[-1]):
            raise ValueError("interpolation point outside exported grid")
        i = int(np.clip(np.floor((x-self.x[0])/self.dx), 1, len(self.x)-3))
        j = int(np.clip(np.floor((y-self.y[0])/self.dy), 1, len(self.y)-3))
        wx, gx, hx = weights((x-self.x[i])/self.dx)
        wy, gy, hy = weights((y-self.y[j])/self.dy)
        block = self.values[i-1:i+3, j-1:j+3]
        if not np.all(np.isfinite(block)):
            raise ValueError("nonfinite interpolation stencil")
        value = float(wx @ block @ wy)
        grad = np.array([gx @ block @ wy/self.dx, wx @ block @ gy/self.dy])
        mixed = float(gx @ block @ gy/(self.dx*self.dy))
        hess = np.array([[hx @ block @ wy/self.dx**2, mixed],
                         [mixed, wx @ block @ hy/self.dy**2]], dtype=float)
        return value, grad, hess


class MaterialPolygon:
    def __init__(self, theta, points):
        theta, points = np.asarray(theta), np.asarray(points)
        if len(theta) > 2 and abs((theta[-1]-theta[0])-TWOPI) < 1e-10:
            theta, points = theta[:-1], points[:-1]
        order = np.argsort(theta % TWOPI)
        self.theta = (theta[order] % TWOPI).astype(float)
        self.points = points[order].astype(float)
        assert self.points.shape == (len(self.theta), 2)
        assert np.all(np.isfinite(self.points))
        self.n = len(theta)
        self.step = TWOPI/self.n
        diffs = np.diff(np.r_[self.theta, self.theta[0]+TWOPI])
        if not np.allclose(diffs, self.step, rtol=1e-8, atol=1e-11):
            raise ValueError("material labels must have uniform periodic theta spacing")
        self.ends = np.roll(self.points, -1, axis=0)
        self.edges = self.ends-self.points
        self.length2 = np.sum(self.edges*self.edges, axis=1)
        if np.any(self.length2 <= 0):
            raise ValueError("degenerate boundary polygon edge")
        area2 = np.sum(self.points[:, 0]*self.ends[:, 1]
                       -self.points[:, 1]*self.ends[:, 0])
        if area2 <= 0:
            raise ValueError("boundary polygon must be counterclockwise")
        # A first-exit map needs a simple domain boundary. Reject strict
        # intersections of nonadjacent polygon edges, rather than trusting area.
        r = self.edges[:, None, :]
        s = self.edges[None, :, :]
        offset = self.points[None, :, :]-self.points[:, None, :]
        denom = r[..., 0]*s[..., 1]-r[..., 1]*s[..., 0]
        mask = abs(denom) > 1e-15
        first = np.zeros_like(denom)
        second = np.zeros_like(denom)
        np.divide(offset[..., 0]*s[..., 1]-offset[..., 1]*s[..., 0], denom,
                  out=first, where=mask)
        np.divide(offset[..., 0]*r[..., 1]-offset[..., 1]*r[..., 0], denom,
                  out=second, where=mask)
        crossing = mask & (first > 1e-10) & (first < 1-1e-10) & (second > 1e-10) & (second < 1-1e-10)
        if np.any(crossing):
            raise ValueError("self-intersecting material boundary polygon")

    def anchor(self, theta):
        coordinate = ((theta-self.theta[0]) % TWOPI)/self.step
        j = int(np.floor(coordinate))
        w, _, _ = weights(coordinate-j)
        return w @ self.points[(j+np.array([-1, 0, 1, 2])) % self.n]

    def inside(self, p):
        x, y = p
        ax, ay = self.points[:, 0], self.points[:, 1]
        bx, by = self.ends[:, 0], self.ends[:, 1]
        crossing = (ay > y) != (by > y)
        indices = np.nonzero(crossing)[0]
        if not len(indices):
            return False
        intercept = ax[indices]+(y-ay[indices])*(bx[indices]-ax[indices])/(by[indices]-ay[indices])
        return bool(np.count_nonzero(x < intercept) % 2)

    def nearest_label(self, p):
        fraction = np.clip(np.sum((p-self.points)*self.edges, axis=1)/self.length2, 0, 1)
        projected = self.points+fraction[:, None]*self.edges
        distances = np.linalg.norm(projected-p, axis=1)
        index = int(np.argmin(distances))
        theta = (self.theta[index]+fraction[index]*self.step) % TWOPI
        return float(theta), projected[index], float(distances[index])

    def chord_first_exit(self, p, q):
        direction = q-p
        edge = self.edges
        offset = self.points-p
        denom = direction[0]*edge[:, 1]-direction[1]*edge[:, 0]
        good = abs(denom) > 1e-15
        u = np.full(self.n, np.inf)
        v = np.full(self.n, np.inf)
        u[good] = (offset[good, 0]*edge[good, 1]-offset[good, 1]*edge[good, 0])/denom[good]
        v[good] = (offset[good, 0]*direction[1]-offset[good, 1]*direction[0])/denom[good]
        # For a CCW polygon an outgoing crossing has cross(edge,direction)<0.
        outgoing = edge[:, 0]*direction[1]-edge[:, 1]*direction[0] < 0
        good &= outgoing & (u > 1e-10) & (u <= 1+1e-10) & (v >= -1e-10) & (v <= 1+1e-10)
        if not np.any(good):
            return None
        index = int(np.argmin(np.where(good, u, np.inf)))
        return float(u[index])


def critical_point(field, epsilon):
    # A moving four-node Lagrange stencil is only C0 across grid lines.
    # Newton on its jumping derivative can oscillate around an exactly symmetric
    # central saddle. Use one fixed symmetric 5x5 polynomial patch for this
    # diagnostic; keep the original scalar/gradient interpolant for tracing.
    i, j = int(np.argmin(abs(field.x))), int(np.argmin(abs(field.y)))
    if min(i, j) < 2 or i+2 >= len(field.x) or j+2 >= len(field.y):
        raise ValueError("central critical diagnostic needs a complete 5x5 stencil")
    center = np.array([field.x[i], field.y[j]])
    inverse = np.linalg.inv(np.vander(np.arange(-2., 3.), 5, increasing=True))
    coefficient = inverse @ field.values[i-2:i+3, j-2:j+3] @ inverse.T
    if not np.all(np.isfinite(coefficient)):
        raise ValueError("nonfinite central polynomial patch")

    def local_jet(p):
        u, v = (p-center)/np.array([field.dx, field.dy])
        wu = np.array([1, u, u*u, u**3, u**4])
        wv = np.array([1, v, v*v, v**3, v**4])
        du = np.array([0, 1, 2*u, 3*u*u, 4*u**3])/field.dx
        dv = np.array([0, 1, 2*v, 3*v*v, 4*v**3])/field.dy
        ddu = np.array([0, 0, 2, 6*u, 12*u*u])/field.dx**2
        ddv = np.array([0, 0, 2, 6*v, 12*v*v])/field.dy**2
        grad = np.array([du @ coefficient @ wv, wu @ coefficient @ dv])
        mixed = du @ coefficient @ dv
        hess = np.array([[ddu @ coefficient @ wv, mixed],
                         [mixed, wu @ coefficient @ ddv]])
        return float(wu @ coefficient @ wv), grad, hess

    point = np.zeros(2)
    for _ in range(20):
        value, grad, hess = local_jet(point)
        if np.linalg.norm(grad) <= epsilon*1e-11:
            break
        shift = np.linalg.solve(hess, grad)
        if np.linalg.norm(shift) > 0.1:
            raise ValueError("critical-point Newton step leaves central neighborhood")
        point -= shift
    value, grad, hess = local_jet(point)
    eigen = np.linalg.eigvalsh(hess)
    return point, value, grad, eigen


def trace(field, polygon, theta, epsilon, saddle_value, step,
          boundary_error, critical_guard=1e-3, max_length=12.0):
    label_c = float(np.sin(2*theta)/2)
    inlet = "N" if np.sin(theta) > 0 else "S"
    point = polygon.anchor(theta)
    initial_phi, _, _ = field.jet(point)
    anchor_error = abs(initial_phi-epsilon*label_c)
    gap = epsilon*label_c-saddle_value
    guard = max(epsilon*critical_guard, 8*(boundary_error+anchor_error))
    result = {"theta_in": float(theta % TWOPI), "c": label_c, "inlet": inlet,
              "gap": float(gap), "guard": float(guard), "anchor_phi_error": float(anchor_error)}
    if abs(gap) <= guard:
        result["status"] = "NEAR_CRITICAL_UNRESOLVED"
        return result
    expected_theta = (np.pi/2-theta+(np.pi if gap < 0 else 0)) % TWOPI
    expected_outlet = "E" if np.cos(expected_theta) > 0 else "W"
    result.update(expected_theta=float(expected_theta), expected_outlet=expected_outlet)
    minimum_grad = np.inf
    maximum_drift = 0.0
    floor = epsilon*1e-9

    def direction(p):
        _, gradient, _ = field.jet(p)
        norm = np.linalg.norm(gradient)
        if not np.isfinite(norm) or norm <= floor:
            raise ValueError("near-critical vanishing gradient")
        return np.array([gradient[1], -gradient[0]])/norm

    def rk4(p, length):
        k1 = direction(p)
        k2 = direction(p+length*k1/2)
        k3 = direction(p+length*k2/2)
        k4 = direction(p+length*k3)
        return p+length*(k1+2*k2+2*k3+k4)/6

    try:
        # The cubic material anchor can sit just outside its polygonal
        # approximation. Enter along the actual interpolated line, never by
        # moving the anchor normal to the field or relabelling it.
        length = 0.0
        for _ in range(20):
            if polygon.inside(point):
                break
            launch = step/8
            point = rk4(point, launch)
            length += launch
        if not polygon.inside(point):
            result["status"] = "UNRESOLVED_INWARD_LAUNCH"
            return result
        for count in range(20000):
            value, gradient, hess = field.jet(point)
            norm = float(np.linalg.norm(gradient))
            minimum_grad = min(minimum_grad, norm)
            maximum_drift = max(maximum_drift, abs(value-initial_phi))
            if maximum_drift > abs(gap)/8:
                result["status"] = "UNRESOLVED_LEVEL_DRIFT"
                break
            if norm <= floor:
                result["status"] = "UNRESOLVED_NEAR_SADDLE"
                break
            local_step = min(step, 0.15*norm/max(float(np.linalg.norm(hess)), epsilon*1e-12))
            if local_step < step*1e-9:
                result["status"] = "UNRESOLVED_SMALL_STEP"
                break
            candidate = rk4(point, local_step)
            crossing = polygon.chord_first_exit(point, candidate)
            if crossing is not None:
                if polygon.inside(candidate):
                    result["status"] = "UNRESOLVED_BOUNDARY_REENTRY"
                    break
                low, high = 0.0, local_step
                for _ in range(35):
                    middle = (low+high)/2
                    if polygon.inside(rk4(point, middle)):
                        low = middle
                    else:
                        high = middle
                endpoint = rk4(point, (low+high)/2)
                theta_out, projection, edge_distance = polygon.nearest_label(endpoint)
                endpoint_phi, _, _ = field.jet(endpoint)
                maximum_drift = max(maximum_drift, abs(endpoint_phi-initial_phi))
                if maximum_drift > abs(gap)/8:
                    result["status"] = "UNRESOLVED_LEVEL_DRIFT"
                    break
                outlet = "E" if np.cos(theta_out) > 0 else "W"
                result.update(status="EXIT", theta_out=theta_out, outlet=outlet,
                              pair_matches=bool(outlet == expected_outlet),
                              label_error=wrap_error(theta_out, expected_theta),
                              endpoint=projection.tolist(), polygon_residual=edge_distance,
                              steps=count+1, arclength=length+(low+high)/2)
                break
            if not polygon.inside(candidate):
                result["status"] = "UNRESOLVED_MISSED_BOUNDARY_CROSSING"
                break
            point = candidate
            length += local_step
            if length > max_length:
                result["status"] = "UNRESOLVED_LENGTH_LIMIT"
                break
        else:
            result["status"] = "UNRESOLVED_STEP_LIMIT"
    except (ValueError, np.linalg.LinAlgError) as error:
        result.update(status="UNRESOLVED_INTERPOLATION_OR_GRADIENT", detail=str(error))
    result.update(minimum_gradient=float(minimum_grad) if np.isfinite(minimum_grad) else None,
                  level_drift=float(maximum_drift))
    return result


def audit(path, step_scale=0.5, critical_guard=1e-3):
    start_time = time.perf_counter()
    with np.load(path, allow_pickle=False) as data:
        x, y = data["x"], data["y"]
        targets = data["gaussian_slices_targets"]
        fields = data["phi_slices"]
        boundary = data["boundary_xy"]
        theta_key = "boundary_theta" if "boundary_theta" in data else "boundary_labels"
        theta = data[theta_key]
        epsilon_key = "epsilon" if "epsilon" in data else "eps"
        epsilon = float(data[epsilon_key])
        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be a finite nonnegative scalar")
        supplied_phi = data["boundary_phi"] if "boundary_phi" in data else None
        supplied_s = data["boundary_s"] if "boundary_s" in data else None
        s_error = data["boundary_s_reconstruction_error"] if "boundary_s_reconstruction_error" in data else None
        assert fields.shape == (len(targets), len(x), len(y))
        assert boundary.shape[:2] == (len(targets), len(theta))
        report = {"source": Path(path).name, "source_case": Path(path).parent.name, "epsilon": epsilon,
                  "step_scale": step_scale, "critical_guard_relative": critical_guard,
                  "scope": "numerical first-exit evidence, not a topology or export certificate", "slices": []}
        for index, target in enumerate(targets):
            if epsilon == 0:
                inner = x[:, None]**2+y[None, :]**2 <= 1.25**2
                if not np.all(np.isfinite(fields[index][inner])):
                    raise ValueError("zero-amplitude control lacks finite data in the diagnostic disk")
                maximum = float(np.max(abs(fields[index][inner])))
                if maximum > 1e-12:
                    raise ValueError("zero-amplitude archive contains a nonzero scalar field")
                report["slices"].append({"s": float(target), "status": "DEGENERATE_ZERO_FIELD",
                    "max_abs_phi": maximum, "traced_exits": 0, "unresolved": 14,
                    "eligible_cases": 0, "unresolved_eligible_cases": 0,
                    "mismatched_pairs": 0, "observed_pair_counts": {},
                    "boundary_phi_error_interpolated": maximum,
                    "max_label_error": None, "max_level_drift": None, "cases": [],
                    "scope": "No nonzero field sector or endpoint map at epsilon=0"})
                continue
            field = CubicGrid(x, y, fields[index])
            polygon = MaterialPolygon(theta, boundary[index])
            p, a, gradient, eigenvalues = critical_point(field, epsilon)
            if np.linalg.norm(gradient) > epsilon*1e-8 or not (eigenvalues[0] < 0 < eigenvalues[1]):
                raise ValueError("central critical point is unresolved or lacks a saddle signature")
            expected_boundary = epsilon*np.sin(2*theta)/2
            boundary_values = np.array([field.jet(v)[0] for v in boundary[index]])
            boundary_error = float(np.max(abs(boundary_values-expected_boundary)))
            slice_result = {"s": float(target), "critical_xy": p.tolist(), "a": float(a),
                            "critical_gradient": float(np.linalg.norm(gradient)),
                            "critical_diagnostic": "fixed central symmetric 5x5 quartic patch",
                            "hessian_eigenvalues": eigenvalues.tolist(),
                            "sampled_morse_signature": bool(eigenvalues[0] < 0 < eigenvalues[1]),
                            "boundary_phi_error_interpolated": boundary_error}
            if supplied_phi is not None:
                slice_result["boundary_phi_error_exported"] = float(np.max(abs(supplied_phi[index]-expected_boundary)))
                slice_result["slice_boundary_phi_disagreement"] = float(np.max(abs(boundary_values-supplied_phi[index])))
            if supplied_s is not None:
                slice_result["boundary_s_error_exported"] = float(np.max(abs(supplied_s[index]-target)))
            if s_error is not None:
                slice_result["boundary_s_inverse_residual_exported"] = float(np.max(abs(s_error[index])))
            if "T_slices" in data:
                Ts = data["T_slices"][index]
                center_i, center_j = int(np.argmin(abs(x))), int(np.argmin(abs(y)))
                slice_result["harmonic_T_at_center"] = float(Ts[center_i, center_j])
            cases = []
            for c in (0.0, -0.04, 0.04, -0.12, 0.12, -0.18, 0.18):
                north = np.pi/2-np.arcsin(2*c)/2
                for angle in (north, north+np.pi):
                    cases.append(trace(field, polygon, angle, epsilon, a,
                                       step_scale*min(field.dx, field.dy), boundary_error,
                                       critical_guard=critical_guard))
            counts = {}
            for case in cases:
                if case["status"] == "EXIT":
                    key = case["inlet"]+"->"+case["outlet"]
                    counts[key] = counts.get(key, 0)+1
            exits = [case for case in cases if case["status"] == "EXIT"]
            eligible = [case for case in cases if abs(case["gap"]) > case["guard"]]
            slice_result.update(cases=cases, observed_pair_counts=counts,
                                traced_exits=len(exits), unresolved=len(cases)-len(exits),
                                eligible_cases=len(eligible),
                                unresolved_eligible_cases=sum(case["status"] != "EXIT" for case in eligible),
                                mismatched_pairs=sum(not case["pair_matches"] for case in exits),
                                max_label_error=max((case["label_error"] for case in exits), default=None),
                                max_level_drift=max((case["level_drift"] for case in exits), default=None))
            report["slices"].append(slice_result)
        report["runtime_seconds"] = time.perf_counter()-start_time
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--step-scale", type=float, default=0.5)
    parser.add_argument("--critical-guard", type=float, default=1e-3)
    args = parser.parse_args()
    report = audit(args.archive, args.step_scale, args.critical_guard)
    output = args.output or args.archive.with_name(args.archive.stem+"_slice_audit.json")
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for result in report["slices"]:
        print(f"s={result['s']:.6g}: exits={result['traced_exits']}, "
              f"unresolved={result['unresolved']}, unresolved_eligible={result['unresolved_eligible_cases']}, "
              f"pair_mismatches={result['mismatched_pairs']}, "
              f"boundary_phi_error={result['boundary_phi_error_interpolated']:.3e}, "
              f"counts={result['observed_pair_counts']}")
    print("Numerical tracing completed; unresolved cases are not classified as no change.")
    print(output)


if __name__ == "__main__":
    main()
