package ui;

import javax.swing.*;

import patientDataLoad.path_loader;

import java.awt.*;
import java.awt.event.*;
import patientDataUtils.*;

public class gui extends JPanel implements ActionListener{
    private JLabel label;
    private JTextField tf;
    private JButton button;

    public static String load_path;

    public gui () {

        // define all panels for grid
        JPanel topPanel = new JPanel();
        JPanel midPanel = new JPanel();
        JPanel bottomPanel = new JPanel();
        JPanel footPanel = new JPanel();

        topPanel.setBorder(BorderFactory.createEtchedBorder());
        midPanel.setBorder(BorderFactory.createEtchedBorder());
        bottomPanel.setBorder(BorderFactory.createEtchedBorder());
        footPanel.setBorder(BorderFactory.createEtchedBorder());

        label = new JLabel("Enter the load directory or enter a single block");
        topPanel.add(label);

        JLabel dir_label = new JLabel("Load directory:");
        midPanel.add(dir_label);

        tf = new JTextField (20);
        midPanel.add(tf);

        button = new JButton ("Enter");
        button.addActionListener(this);
        midPanel.add(button);

        block_entry(bottomPanel);

        JButton plus_button = new JButton("+");
        plus_button.addActionListener(this);
        footPanel.add(plus_button);

        BoxLayout boxLayout = new BoxLayout(this, BoxLayout.PAGE_AXIS);
        setLayout(boxLayout);
        add(topPanel);
        add(midPanel);
        add(bottomPanel);
        add(footPanel);

    }

    public static void createAndShowGUI () {

        JFrame.setDefaultLookAndFeelDecorated(true);

        // create frame and add icon
        JFrame frame = new JFrame();
        Image icon = Toolkit.getDefaultToolkit().getImage("../data/gradient_icon.png");
        frame.setIconImage(icon);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // add panel with elements
        JPanel panel = new gui();
        frame.add(panel);
        frame.pack();
        frame.setVisible(true);
    }

    public static void main(String[] args) {
        
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                createAndShowGUI();
            }
        });
    }

    public void actionPerformed (ActionEvent e) {
        if (e.getSource() == button) {
            load_path = tf.getText();
            path_loader loader = new path_loader(load_path);
        }
    }

    // method to make individual block entries 
    public void block_entry (JPanel panel) {
        label = new JLabel("Enter the label for this block:");
        JTextField tf = new JTextField (20);
        panel.add(label);
        panel.add(tf);

        label = new JLabel("Enter the path for an image:");
        JTextField data_path_entry = new JTextField(20);
        panel.add(label);
        panel.add(data_path_entry);

        JButton blockButton = new JButton ("Enter");
        blockButton.addActionListener(this);
        panel.add(blockButton);
    }
}
