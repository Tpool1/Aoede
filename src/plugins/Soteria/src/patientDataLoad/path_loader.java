package patientDataLoad;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import patientDataUtils.image;

public class path_loader {
   public path_loader(String load_path) {
       String[] filesArr = getFiles(load_path);
       image[] imgArr = loadFiles(filesArr);
       System.out.println(Arrays.toString(imgArr));
   }

   public static String[] getFiles(String directory) {
       File dir = new File(directory);

       Collection<String> files = new ArrayList<String>();
       if (dir.isDirectory()) {
           File[] listFiles = dir.listFiles();

           for (File file : listFiles) {
               if (file.isFile()) {
                   files.add(file.getName());
               }
           }
       }

       return files.toArray(new String[]{});
   }

   public static image[] loadFiles(String[] fileArr) {
       // make array for image objs with same length as fileArr
       image[] imgArr = new image[fileArr.length];

       for (int i = 0; i < fileArr.length; i++) {
           String path = fileArr[i];
           image path_img = new image(path);
           imgArr[i] = path_img;
       }

       return imgArr;
   }
}
