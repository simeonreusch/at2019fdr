#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging, os, argparse, time, glob, json, warnings
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy import units as u
from astropy.time import Time


RA_TRUE = 257.278553
DEC_TRUE = 26.855697
RADEC_REFSTAR = [257.27824, 26.86784]
PIX_REFSTAR = [1355, 1477]

bands = ["J", "H", "Ks"]
apertures = [3, 5, 7, 9]


def source_extractor(band, directory, aper=None):
    """Run source extractor"""
    infile = os.path.join(directory, f"ZTF19aatubsj_{band}_stack_1.fits")
    param_file = os.path.join(directory, "default.param")
    conv_file = os.path.join(directory, "default.conv")
    if aper is None:
        config_file = os.path.join(directory, "default.sex")
        outfile = os.path.join(directory, f"{band}.cat")
    else:
        config_file = os.path.join(directory, f"default_aper_{aper}.sex")
        outfile = os.path.join(directory, f"{band}_aper_{aper}.cat")

    command = [
        f"source-extractor {infile} -c {config_file} -CATALOG_NAME {outfile} -PARAMETERS_NAME {param_file} -FILTER_NAME {conv_file}"
    ]
    s0 = ""
    command = s0.join(command)
    print(command)
    res = os.system(command)
    return res


def flux_to_mag(flux):
    return -2.5 * np.log10(flux)


def flux_to_abmag(fluxnu):
    return (-2.5 * np.log10(fluxnu)) - 48.585


def transform_pixels_to_wcs(fits_file, pixelpairs):
    hdulist = fits.open(fits_file)
    w = wcs.WCS(hdulist[0].header)
    pixcrd = np.array(pixelpairs, dtype=np.float64)
    world = w.wcs_pix2world(pixcrd, 0)
    return world


def extract_magnitude(
    sextractor_filename,
    fits_filename,
    ZP_aper,
    ZP_aper_err,
    ZP_2MASS,
    ZP_2MASS_err,
    radec_refstar,
):
    sextractor_file = os.path.join(DATA_DIR, sextractor_filename)
    fits_file = os.path.join(DATA_DIR, fits_filename)

    df = pd.read_table(
        sextractor_file,
        names=[
            "X_IMAGE",
            "Y_IMAGE",
            "MAG_BEST",
            "MAGERR_BEST",
            "MAG_APER",
            "MAGERR_APER",
            "MAG_AUTO",
            "MAGERR_AUTO",
            "MAG_ISO",
            "MAGERR_ISO",
        ],
        sep="\s+",
        comment="#",
    )

    pixelpairs = []
    x_pixels = df["X_IMAGE"].values
    y_pixels = df["Y_IMAGE"].values
    x_y_combined = np.asarray(list(zip(x_pixels, y_pixels)))
    for item in x_y_combined:
        pixelpairs.append(list(item))

    transformed_pixels = transform_pixels_to_wcs(fits_file, pixelpairs)

    ras = []
    decs = []
    for coords in transformed_pixels:
        ra = coords[0]
        dec = coords[1]
        ras.append(ra)
        decs.append(dec)

    df["ra"] = np.asarray(ras)
    df["dec"] = np.asarray(decs)

    true_position = SkyCoord(RA_TRUE * u.deg, DEC_TRUE * u.deg, frame="icrs")
    position_refstar = SkyCoord(
        RADEC_REFSTAR[0] * u.deg, RADEC_REFSTAR[1] * u.deg, frame="icrs"
    )
    distances = []
    distances_refstar = []
    for index, ra in enumerate(df.ra.values):
        object_pos = SkyCoord(ra * u.deg, df.dec.values[index] * u.deg, frame="icrs")
        dist = true_position.separation(object_pos)
        dist_refstar = position_refstar.separation(object_pos)
        distances.append(dist.arcsecond)
        distances_refstar.append(dist_refstar)

    df["distances"] = distances
    df["distances_refstar"] = distances_refstar
    df.sort_values(by="distances", inplace=True)
    df.reset_index(inplace=True)

    mag_kron = df.MAG_AUTO.values[0] + ZP_2MASS
    magerr_kron = np.sqrt(df.MAGERR_AUTO[0] ** 2 + ZP_2MASS_err ** 2)
    mag_iso = df.MAG_ISO.values[0] + ZP_2MASS
    magerr_iso = np.sqrt(df.MAGERR_ISO[0] ** 2 + ZP_2MASS_err ** 2)
    mag_aper = df.MAG_APER.values[0] + ZP_aper
    magerr_aper = np.sqrt(df.MAGERR_APER[0] ** 2 + ZP_aper_err ** 2)

    df.sort_values(by="distances_refstar", inplace=True)
    df.reset_index(inplace=True)

    refstar_mag_aper = df.MAG_APER.values[0] + ZP_aper
    refstar_mag_iso = df.MAG_ISO.values[0] + ZP_2MASS
    refstar_mag_kron = df.MAG_AUTO.values[0] + ZP_2MASS

    returndict = {
        "mag_kron": mag_kron,
        "magerr_kron": magerr_kron,
        "mag_iso": mag_iso,
        "magerr_iso": magerr_iso,
        "mag_aper": mag_aper,
        "magerr_aper": magerr_aper,
        "refstar_mag_aper": refstar_mag_aper,
        "refstar_mag_iso": refstar_mag_iso,
        "refstar_mag_kron": refstar_mag_kron,
    }

    return returndict


if __name__ == "__main__":

    DATES = ["2020_07_01", "2020_09_29", "2021_02_04"]
    DATES_ISO = [date.replace("_", "-") + "T00:00:00" for date in DATES]
    DATES_MJD = [
        Time(date_iso, format="isot", scale="utc").mjd for date_iso in DATES_ISO
    ]

    parser = argparse.ArgumentParser(
        description="Run source-extractor on tywin NIR-images"
    )
    parser.add_argument(
        "-extract", action="store_true", help="(Re-) run the actual extraction"
    )
    commandline_args = parser.parse_args()
    do_extraction = commandline_args.extract

    df = pd.DataFrame(index=DATES_MJD)

    for i, DATE in enumerate(DATES):

        DATA_DIR = os.path.join("data", f"P200_NIR_observations_{DATE}")

        if do_extraction:
            for band in ["J", "H", "Ks"]:
                for aper in apertures:
                    source_extractor(band, DATA_DIR, aper)

        zeropoints_aper = {}
        zeropoints_2mass = {}
        for file in glob.glob(f"{DATA_DIR}/*.fits"):
            hdul = fits.open(file)
            header = hdul[0].header
            band = header["AFT"].split("_")[0]
            zp_2mass = float(header["TMC_ZP"])
            zp_2mass_err = float(header["TMC_ZPSD"])
            zp_aper = {}
            for aper in apertures:
                zp_aper.update(
                    {
                        aper: [
                            float(header[f"ZP_{aper}"]),
                            float(header[f"ZPSTD_{aper}"]),
                        ]
                    }
                )
            zeropoints_aper.update({band: zp_aper})
            zeropoints_2mass.update({band: [zp_2mass, zp_2mass_err]})

        sextractor_files = ["J.cat", "H.cat", "Ks.cat"]
        fits_files = [
            "ZTF19aatubsj_J_stack_1.fits",
            "ZTF19aatubsj_H_stack_1.fits",
            "ZTF19aatubsj_Ks_stack_1.fits",
        ]

        bands = ["J", "H", "Ks"]

        mags = {}

        for k, band in enumerate(bands):
            for aper in apertures:
                sextractor_outfile = f"{band}_aper_{aper}.cat"
                fits_file = fits_files[k]
                result = extract_magnitude(
                    sextractor_outfile,
                    fits_file,
                    zeropoints_aper[band][aper][0],
                    zeropoints_aper[band][aper][1],
                    zeropoints_2mass[band][0],
                    zeropoints_2mass[band][1],
                    RADEC_REFSTAR,
                )

                df.at[DATES_MJD[i], f"{band}_mag_kron"] = result["mag_kron"]
                df.at[DATES_MJD[i], f"{band}_mag_kron_err"] = result["magerr_kron"]
                df.at[DATES_MJD[i], f"{band}_mag_iso"] = result["mag_iso"]
                df.at[DATES_MJD[i], f"{band}_mag_iso_err"] = result["magerr_iso"]
                df.at[DATES_MJD[i], f"{band}_mag_aper_{aper}"] = result["mag_aper"]
                df.at[DATES_MJD[i], f"{band}_mag_aper_{aper}_err"] = result[
                    "magerr_aper"
                ]
                df.at[DATES_MJD[i], f"{band}_refstar_mag_aper_{aper}"] = result[
                    "refstar_mag_aper"
                ]
                df.at[DATES_MJD[i], f"{band}_refstar_mag_iso"] = result[
                    "refstar_mag_iso"
                ]
                df.at[DATES_MJD[i], f"{band}_refstar_mag_kron"] = result[
                    "refstar_mag_kron"
                ]

    df.to_csv(os.path.join("data", "P200_unsubtracted.csv"), index_label="obsmjd")
