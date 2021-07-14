import java.util.*;
import patientDataUtils.*;
import ui.gui;

public class Main {
    public static ArrayList<Block> blockchain = new ArrayList<Block>();
    public static int difficulty = 5;

    public static void main(String[] args) {

        gui.createAndShowGUI();

        image img = new image("../data/portrait.jpeg");
        dicom_data patient1 = new dicom_data(1, img);

        Block genesis = makeGenesis(patient1);
        makeBlock(1, genesis.currentHash, "second block", patient1);
    }

    public static Block makeGenesis(dicom_data patient) {
        // make genesis block
        Block b = new Block(0, null, "genesis", patient);
        b.mineBlock(difficulty);
        blockchain.add(b);
        System.out.println(b.toString());
        return b;
    }

    public static void makeBlock(int index, String previousHash , String label, dicom_data patient) {
        // make second block
        Block b = new Block(index, previousHash, label, patient);
        b.mineBlock(difficulty);
        blockchain.add(b);
        System.out.println(b.toString());
    }
} 
