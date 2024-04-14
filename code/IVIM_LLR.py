#!/bin/python3

# Author: Alan Finkelstein
import datetime
import numpy as np
from natsort import natsorted
import argparse
import cfl
from utils import *
import mat73
import os
import nibabel as nib
import scipy.io as sio

description = "Improved IVIM Parameter Estimation Using Locally Low Rank and Subspace Constraints.\n" \
              "Correction of shot-to-shot phase variations in IVIM data."


def parser():
    parse = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument("--data", type=str)
    parse.add_argument("--sens", type=str)
    parse.add_argument("--outdir", type=str)
    parse.add_argument("--basis_size", type=int, default=2)  # also do with 2 and 3 and 4 see if there is a difference.
    parse.add_argument("--subspace", type=bool, default=True, help="Whether or not to use subspace")
    parse.add_argument("--lowres", type=bool, default=False)
    parse.add_argument("--LLR", type=bool, default=False)
    parse.add_argument("--method", type=str, default='lst',
                       help='Method can be DL, bayes, or lst (for segmented least squares)')
    parse.add_argument("--lenb", type=int, default=15, help="Can be 15 for full or 8 for partial")
    parse.add_argument("--get_ksp", type=str, default="False")
    parse.add_argument("--EPI_factor", type=int, default=2)
    parse.add_argument("--lambda1", type=float, default=0.001, help="LLR reg") # try 0.001, 0.005, and 0.0005, and 0.0001
    parse.add_argument("--lambda2", type=float, default=0.001, help="TV reg")
    parse.add_argument("--slice", type=int, default=34)
    parse.add_argument("--vis", type=str, default="False")
    parse.add_argument("--local", type=bool, default=False)

    return parse.parse_args()

def main():
    args = parser()
    if os.path.exists(args.outdir):
        pass
    else:
        os.mkdir(args.outdir)

    print("Running XYZ Trace!!!")
    print(args.outdir, flush=True)
    if args.get_ksp == "True":
        print("Now getting kspace data...")
        data = np.zeros((164 * 2, 164, 16, 35, 67), dtype=np.complex128)
        for i, img in enumerate(natsorted(os.listdir(args.data))):
            try:
                data[..., i] = mat73.loadmat(os.path.join(args.data, img))['kspace_full']
            except TypeError:
                data[..., i] = sio.loadmat(os.path.join(args.data, img))['kspace_full']
        nib.save(nib.Nifti1Image(data, np.eye(4)), os.path.join(args.outdir, 'data_kspace.nii.gz'))
    else:
        print("Loading ksp data...")
        data = nib.load(os.path.join(args.outdir, 'data_kspace.nii.gz')).get_fdata(dtype=np.complex64)
        print("Done loading ksp data..")
    if args.local:
        tmp_dir = args.outdir.split('/')
        tmp_dir2 = os.path.join(*tmp_dir[:tmp_dir.index('dwi')+1])
        sub, timept = tmp_dir[tmp_dir.index('dwi')-2],  tmp_dir[tmp_dir.index('dwi')-1]
    else:
        tmp_dir = args.outdir.split('/')
        tmp_dir2 = os.path.join(*tmp_dir[:-1])
        sub, timept = tmp_dir[tmp_dir.index('data') +1], tmp_dir[tmp_dir.index('data') +2]

    affine = nib.load(os.path.join('/' +tmp_dir2,'{}_{}_enc-ap_ivim.nii.gz'.format(sub, timept))).affine
    shift_amount = int(-np.ceil(164 / (2 * 2)))  # CAIPI shift correction

    basis = cfl.readcfl('Standard_Files/ivim_basis_{}'.format(args.basis_size))
    recon = np.zeros((int(data.shape[0] / 2), data.shape[1], data.shape[3] * 2, 15),
                     dtype=np.complex64)  # SUBSPACE Recon
    reconx, recony, reconz = np.zeros_like(recon), np.zeros_like(recon), np.zeros_like(recon)

    recon_sense = np.zeros((int(data.shape[0] / 2), data.shape[1], data.shape[3] * 2, 15),
                           dtype=np.complex64)  # SENSE recon
    recon_sensex, recon_sensey, recon_sensez = np.zeros_like(recon), np.zeros_like(recon), np.zeros_like(recon)

    for sli in range(data.shape[3]):  # data.shape[35]):  # loop over slices, 35 slices x 2 (SMS, slice acceleration factor).
        print("CURRENTLY RECONSTRUCTING SLICE: {}".format(sli), flush=True)
        # Extended FOV with CAIPI shifted for SMS reconstruction.
        data_slice = data[:, :, :, sli, :]
        data_slice = np.expand_dims(data_slice[..., 1:46], axis=2)
        # sens shape is x (368 (SMS)) * y (164)  * 1 * coil (16)
        sens = np.transpose(np.expand_dims(mat73.loadmat(args.sens)['sensitivity'][:, :, :, sli], axis=3),
                            (0, 1, 3, 2))
        sens_prelim = get_initial_sens(data_slice, sens)  # 328x164x15
        sens_prelim2 = np.mean([np.abs(sens_prelim.squeeze()[..., ::3]), np.abs(sens_prelim.squeeze()[..., 1::3]),
                                np.abs(sens_prelim.squeeze()[..., 2::3])], axis=0)
        recon_sense[:, :, sli, :] = sens_prelim2[:164, :, :]
        recon_sense[:, :, sli + 35, :] = np.roll(sens_prelim2[164:, :, :], shift_amount, axis=1)

        x_sens, phase_estimates = get_composite_sens(sens_prelim[..., ::3], sens, visualize="False", title='x')
        x_sens = np.expand_dims(np.transpose(x_sens, (0, 1, 4, 2, 3)), axis=4)
        y_sens, phase_estimates = get_composite_sens(sens_prelim[..., 1::3], sens, visualize="False", title='y')

        y_sens = np.expand_dims(np.transpose(y_sens, (0, 1, 4, 2, 3)), axis=4)
        z_sens, _ = get_composite_sens(sens_prelim[..., 2::3], sens, visualize="False", title='z')
        z_sens = np.expand_dims(np.transpose(z_sens, (0, 1, 4, 2, 3)), axis=4)

        data_slice = np.expand_dims(data_slice, axis=4)
        data_slice_x = data_slice[..., ::3]
        data_slice_y = data_slice[..., 1::3]
        data_slice_z = data_slice[..., 2::3]

        recontmp_x, recon_fmac_x = llr_recon(data_slice_x, x_sens, basis,
                                             use_basis=True, R=args.EPI_factor, lambda1=args.lambda1,
                                             lambda2=args.lambda2)

        recontmp_y, recon_fmac_y = llr_recon(data_slice_y, y_sens, basis,
                                             use_basis=True, R=args.EPI_factor, lambda1=args.lambda1,
                                             lambda2=args.lambda2)

        recontmp_z, recon_fmac_z = llr_recon(data_slice_z, z_sens, basis,
                                             use_basis=True, R=args.EPI_factor, lambda1=args.lambda1,
                                             lambda2=args.lambda2)

        recon_fmac_x, recon_fmac_y, recon_fmac_z = np.real(recon_fmac_x.squeeze()), np.real(
            recon_fmac_y.squeeze()), np.real(recon_fmac_z.squeeze())
        recon_fmac = np.mean([np.real(recon_fmac_x), np.real(recon_fmac_y), np.real(recon_fmac_z)], axis=0)
        recon_fmac = recon_fmac.squeeze()

        recon[:, :, sli, :] = recon_fmac[:164, :, :]
        recon[:, :, sli + 35, :] = np.roll(recon_fmac[164:, :, :], shift_amount, axis=1)

        reconx[:, :, sli, :] = recon_fmac_x[:164, :, :]
        reconx[:, :, sli + 35, :] = np.roll(recon_fmac_x[164:, :, :], shift_amount, axis=1)

        recony[:, :, sli, :] = recon_fmac_y[:164, :, :]
        recony[:, :, sli + 35, :] = np.roll(recon_fmac_y[164:, :, :], shift_amount, axis=1)

        reconz[:, :, sli, :] = recon_fmac_z[:164, :, :]
        reconz[:, :, sli + 35, :] = np.roll(recon_fmac_z[164:, :, :], shift_amount, axis=1)

        recon_sense[:, :, sli, :] = sens_prelim2[:164, :, :]
        recon_sense[:, :, sli + 35, :] = np.roll(sens_prelim2[164:, :, :], shift_amount, axis=1)

        recon_sensex[:, :, sli, :] = sens_prelim[:164, :, ::3]
        recon_sensex[:, :, sli + 35, :] = np.roll(sens_prelim[164:, :, ::3], shift_amount, axis=1)

        recon_sensey[:, :, sli, :] = sens_prelim[:164, :, 1::3]
        recon_sensey[:, :, sli + 35, :] = np.roll(sens_prelim[164:, :, 1::3], shift_amount, axis=1)

        recon_sensez[:, :, sli, :] = sens_prelim[:164, :, 2::3]
        recon_sensez[:, :, sli + 35, :] = np.roll(sens_prelim[164:, :, 2::3], shift_amount, axis=1)

    nib.save(nib.Nifti1Image(np.real(recon), affine=affine),
             os.path.join(args.outdir, "ivim_recon_subspace_{}.nii.gz".format(args.basis_size)))
    nib.save(nib.Nifti1Image(np.real(reconx), affine=affine),
             os.path.join(args.outdir, "ivim_recon_subspace_{}_x.nii.gz".format(args.basis_size)))
    nib.save(nib.Nifti1Image(np.real(recony), affine=affine),
             os.path.join(args.outdir, "ivim_recon_subspace_{}_y.nii.gz".format(args.basis_size)))
    nib.save(nib.Nifti1Image(np.real(reconz), affine=affine),
             os.path.join(args.outdir, "ivim_recon_subspace_{}_z.nii.gz".format(args.basis_size)))

    nib.save(nib.Nifti1Image(abs(recon_sense), affine=affine), os.path.join(args.outdir, "ivim_recon_sense.nii.gz"))
    nib.save(nib.Nifti1Image(abs(recon_sensex), affine=affine),
             os.path.join(args.outdir, "ivim_recon_sense_x.nii.gz"))
    nib.save(nib.Nifti1Image(abs(recon_sensey), affine=affine),
             os.path.join(args.outdir, "ivim_recon_sense_y.nii.gz"))
    nib.save(nib.Nifti1Image(abs(recon_sensez), affine=affine),
             os.path.join(args.outdir, "ivim_recon_sense_z.nii.gz"))

    return None


if __name__ == "__main__":
    main()
