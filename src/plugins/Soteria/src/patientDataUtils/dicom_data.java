package patientDataUtils;

public class dicom_data{
    public int patient_id;
    public image patient_image;

    public dicom_data(int patient_id, image patient_image) {
        this.patient_id = patient_id;
        this.patient_image = patient_image;
    }
}
