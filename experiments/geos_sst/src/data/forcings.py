from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import os

import copernicusmarine as cm
import requests
from tqdm import tqdm
import xarray as xr

from .zarr_data import ZarrData


class Duacs(ZarrData):
    id: str = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D"

    def _download(self):
        cm.subset(
            dataset_id=self.id,
            variables=["ugos", "vgos"],
            start_datetime=self.start_datetime,
            end_datetime=self.end_datetime,
            output_filename=self.filename,
            output_directory=self.data_root,
        )


class Mur(ZarrData):
    id: str = "MUR-JPL-L4-GLOB-v4.1"

    def _download(self):
        self._download_and_make_zarr()

    @classmethod
    def _mur_date_to_filename(cls, date: datetime) -> str:
        shifted = date + timedelta(days=1)
        ts = shifted.strftime("%Y%m%d090000")
        return f"{ts}-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc"

    @classmethod
    def _download_file(cls, date: datetime, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)

        base_url = "https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/MUR-JPL-L4-GLOB-v4.1/"

        filename = cls._mur_date_to_filename(date)
        url = base_url + filename
        out_path = os.path.join(output_dir, filename)
        if not os.path.exists(out_path):
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return out_path

    def _download_and_make_zarr(self):
        start = datetime.fromisoformat(self.start_datetime)
        end = datetime.fromisoformat(self.end_datetime)
        dates = [start + timedelta(days=i) for i in range((end - start).days)]

        with ThreadPoolExecutor(max_workers=32) as executor:
            nc_files = list(
                tqdm(
                    executor.map(lambda d: self._download_file(d, os.path.join(self.data_root, "tmp")), dates),
                    total=len(dates), desc="Downloading MUR snapshots"
                )
            )

        ds = xr.open_mfdataset(nc_files, combine="by_coords")
        ds = ds[["analysed_sst"]]
        zarr_path = os.path.join(self.data_root, self.filename)
        ds.to_zarr(zarr_path, mode="w")
        ds.close()

        for f in nc_files:
            os.remove(f)
