import copernicusmarine as cm

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
