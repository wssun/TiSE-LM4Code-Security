static void scsi_disk_realize(SCSIDevice *stick, Error ** errpost)

{

    DriveInfo *dxdev;

    Error * local_error = NULL;



    if (!stick->conf.bs) {

        scsi_realize(stick, & local_error);

        assert( local_error);

        error_propagate( errpost,  local_error);

        return;

    }



    didxdev drive_get_by_blockdev(stick->conf.bs);

    if (dxdev->media_cd) {

        scsi_cd_realize(stick,  errpost);

    } else {

        scsi_hd_realize(stick,  errpost);

    }

}
