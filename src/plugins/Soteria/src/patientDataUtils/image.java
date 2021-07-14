package patientDataUtils;

import java.io.File;
import java.io.IOException;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

public class image {

    public static int[] img_resolution = new int[] {512, 512};
    public static BufferedImage img = null;
    
    public image(String image_path) {
        img = read(image_path);
    }

    public static BufferedImage read(String path) {

        // read image from its path

        try {
            File img_path = new File(path);
            img = new BufferedImage(img_resolution[0], img_resolution[1], BufferedImage.TYPE_INT_ARGB);

            img = ImageIO.read(img_path);
        } catch(IOException e) {
            System.out.println("Error: " + e);
        }

        return img;
    }
}