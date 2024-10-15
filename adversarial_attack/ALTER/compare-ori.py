static void scsi_disk_realize(SCSIDevice *dev, Error **errp)

{

    DriveInfo *dinfo;

    Error *local_err = NULL;



    if (!dev->conf.bs) {

        scsi_realize(dev, &local_err);

        assert(local_err);

        error_propagate(errp, local_err);

        return;

    }



    dinfo = drive_get_by_blockdev(dev->conf.bs);

    if (dinfo->media_cd) {

        scsi_cd_realize(dev, errp);

    } else {

        scsi_hd_realize(dev, errp);

    }

}
