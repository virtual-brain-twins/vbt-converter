import argparse

import h5py
import numpy as np

parameters = ['scale_num_neurons', 'scale_syn_strength', 'scale_nmda_efficacy']
parameters_short = ["numneur", "synstr", "nmdaeff"]

populations = [
    "L1VIP",
    "L23E", "L23PV", "L23SST", "L23VIP",
    "L4E", "L4PV", "L4SST", "L4VIP",
    "L5E", "L5PV", "L5SST", "L5VIP",
    "L6E", "L6PV", "L6SST", "L6VIP"
]


def interpolate_constant(x, xp, yp, extrapolate=False):
    if extrapolate:
        yp = np.concatenate(([yp[0]], yp, [yp[-1]]))

    indices = np.searchsorted(xp, x, side='right')
    return np.array(yp)[indices]


def get_discretized_stimulus(stim, dt):
    stim_type = stim.attrs['type']

    if stim_type == 'piecewise_constant':
        limits = np.array(stim['limits']) / 1000  # Convert to seconds
        t0 = limits[0]
        t1 = limits[-1]
        T = t1 - t0
        nbins = round(T / dt)

        # Times correspond to the bin start point.
        # This means that the first time is at t=t0. Last is not t=t1.
        t = np.linspace(t0, t1, nbins + 1)
        tmid = (t[1:] + t[:-1]) / 2.
        u = interpolate_constant(tmid, limits, np.array(stim['values']), extrapolate=True)
        return u

    else:
        raise NotImplementedError(f"stim_type '{stim_type}' not implemented.")


def get_firing_rates(spikes, dt, t0, t1, nneurons):
    T = t1 - t0
    nbins = round(T / dt)
    t = np.linspace(t0, t1, nbins + 1)
    tmid = (t[1:] + t[:-1]) / 2.
    r = np.zeros((len(populations), nbins))

    nneurons = dict(zip([lab.decode('UTF-8') for lab in nneurons['label']], nneurons['value']))

    for i, pop in enumerate(populations):
        counts, _ = np.histogram(spikes[pop]['times'], bins=t)
        r[i, :] = counts
        r[i, :] /= nneurons[pop]

    r /= dt

    return t[:-1], r


def normalize(x, axis):
    mu = np.mean(x, axis=axis)
    sd = np.std(x, axis=axis)
    return (x - mu) / sd, mu, sd


def convert(input_file, output_file, bin_size):
    with h5py.File(input_file, 'r') as h:
        r = []
        u = []
        theta = []

        for name, sim_grp in h["simulations"].items():
            param_values = np.array([float(sim_grp.attrs[p]) for p in parameters])

            _, firing_rate = get_firing_rates(sim_grp['spike_times'], bin_size,
                                              0, sim_grp.attrs['t_sim'] / 1000.,
                                              sim_grp['record_neurons'])
            stim = get_discretized_stimulus(sim_grp['stimulus'], bin_size)

            r.append(firing_rate.T)
            u.append(stim[:, None])
            theta.append(param_values)

        r = np.array(r)  # shape: (npoints, nt, nvar)
        u = np.array(u)  # shape: (npoints, nt, 1)
        theta = np.array(theta)  # shape: (npoints, nparams)

        # Normalize
        r_norm, mu_r, sd_r = normalize(r, axis=(0, 1))
        u_norm, mu_u, sd_u = normalize(u, axis=(0, 1))
        theta_norm, mu_theta, sd_theta = normalize(theta, axis=0)

    np.savez(output_file, x=r_norm, u=u_norm, theta=theta_norm,
             names_x=populations, names_theta=parameters_short, dt=bin_size,
             mu_x=mu_r, sd_x=sd_r,
             mu_u=mu_u, sd_u=sd_u,
             mu_theta=mu_theta, sd_theta=sd_theta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True)
    parser.add_argument('-o', '--output_file', type=str, required=True)
    parser.add_argument('-b', '--bin_size', type=str, required=True)
    args = parser.parse_args()

    convert(args.input_file, args.output_file, float(args.bin_size))


if __name__ == "__main__":
    main()
