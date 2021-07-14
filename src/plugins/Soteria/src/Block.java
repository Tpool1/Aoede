import java.security.*;
import patientDataUtils.*;

public class Block {
    public int index;
    public long timestamp;
    public String currentHash;
    public String previousHash;
    public String label;
    public dicom_data data;
    public int nonce;

    public Block(int index, String previousHash, String label, dicom_data data) {
        this.index = index;
        this.timestamp = System.currentTimeMillis();
        this.label = label;
        this.data = data;
        nonce = 0;
        currentHash = calculateHash();
    }

    public String calculateHash() {
        try {
            String input = index + timestamp + previousHash + label + nonce;
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(input.getBytes("UTF-8"));

            StringBuffer hexString = new StringBuffer();
            for (int i = 0; i < hash.length; i++) {
                String hex = Integer.toHexString(0xff & hash[i]);
                if(hex.length() == 1) hexString.append('0');
                hexString.append(hex);
            }
            return hexString.toString();
        }
        catch(Exception e) {
            throw new RuntimeException(e);
        }
    }

    public void mineBlock(int difficulty) {
        nonce = 0;
        String target = new String(new char[difficulty]).replace('\0', '0');
        while (!currentHash.substring(0, difficulty).equals(target)) {
            nonce++;
            currentHash = calculateHash();
        }
    }

    public String toString() {
        String s = "Block #     : " + index + "\r\n";
         s = s + "PreviousHash     : " + previousHash + "\r\n";
         s = s + "Timestamp     : " + timestamp + "\r\n";
         s = s + "Label          : " + label + "\r\n";
         s = s + "Nonce         : " + nonce + "\r\n";
         s = s + "CurrentHash       : " + currentHash + "\r\n";
    
    return s;
    }
}
