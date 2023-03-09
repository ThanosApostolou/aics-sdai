package com.unipi.talepis.test;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.lang.reflect.*;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;


public class Form1 extends JFrame{
    private JPanel panel1;
    private JButton button1;
    private JComboBox comboBox1;
    private JTextArea textArea1;

    public Form1() {
        setTitle("Reflection Form");
        setPreferredSize(new Dimension(420, 650));
        setContentPane(panel1);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pack();
        setLocationRelativeTo(null);
        setVisible(true);
        button1.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                //System.out.println(Form1.class.getProtectionDomain().getCodeSource().getLocation().getPath());
                //System.out.println(Form1.class.getPackageName());
                //System.out.println(Form1.class.getPackageName().replace('.','/'));
                String targetDir =this.getClass().getProtectionDomain().getCodeSource().getLocation().getPath()+
                        this.getClass().getPackageName().replace('.','/');
                List<String> classNames = listFileNames(targetDir).stream().
                        filter(s -> !s.contains("$")).
                        map(s->s.substring(0,s.indexOf('.')))
                        .collect(Collectors.toList());
                //JOptionPane.showMessageDialog(Form1.this,classNames);
                comboBox1.setModel(new DefaultComboBoxModel(classNames.toArray()));
            }
        });
        comboBox1.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                    textArea1.setText("");
                    List<String> fields = new ArrayList<>();
                    getAllClassFields(fields,Class.forName(
                            this.getClass().getPackageName()+"."+comboBox1.getSelectedItem().toString()));
                    StringBuilder builderTextFields = new StringBuilder();
                    fields.stream().forEach(s->builderTextFields.append(s).append("\n"));
                    List<String> constructors = new ArrayList<>();
                    getAllClassConstructors(constructors,Class.forName(
                            this.getClass().getPackageName()+"."+comboBox1.getSelectedItem().toString()));
                    StringBuilder builderTextConstructors = new StringBuilder();
                    constructors.stream().forEach(s->builderTextConstructors.append(s).append("\n"));
                    List<String> methods = new ArrayList<>();
                    getAllClassMethods(methods,Class.forName(
                            this.getClass().getPackageName()+"."+comboBox1.getSelectedItem().toString()));
                    StringBuilder builderTextMethods = new StringBuilder();
                    methods.stream().forEach(s->builderTextMethods.append(s).append("\n"));
                    textArea1.setText(builderTextFields+
                            "-----------------------------\n"
                            +builderTextConstructors+
                            "-----------------------------\n"
                            +builderTextMethods);
                } catch (ClassNotFoundException classNotFoundException) {
                    classNotFoundException.printStackTrace();
                }
            }
        });
    }
    private static List<String> listFileNames(String directory){
        return Stream.of(new File(directory).listFiles())
                .filter(file -> file.isFile())
                .map(File::getName)
                .collect(Collectors.toList());
    }
    private static void getAllClassFields(List<String> allFields, Class c){
        Field[] fields = c.getFields();
        Field[] declaredFields = c.getDeclaredFields();
        for (Field f :
                fields) {
            allFields.add(Modifier.toString(f.getModifiers())+" "+f.getType().getSimpleName()+
                    " "+f.getName());
        }
        for (Field f :
                declaredFields) {
            String s = Modifier.toString(f.getModifiers())+" "+f.getType().getSimpleName()+
                    " "+f.getName();
            if (!allFields.contains(s))
                allFields.add(s);
        }
        if (c.getSuperclass() != null) {
            getAllClassFields(allFields, c.getSuperclass());
        }
    }
    private static void getAllClassMethods(List<String> allMethods, Class c){
        Method[] methods = c.getMethods();
        Method[] declaredMethods = c.getDeclaredMethods();
        for (Method f :
                methods) {
            Parameter[] parameterNames = f.getParameters();
            Class[] parameterTypes = f.getParameterTypes();
            StringBuilder parameterBuilder = new StringBuilder();
            for (int i=0;i<parameterNames.length;i++){
                parameterBuilder.append(parameterTypes[i].getSimpleName()).append(" ").append(parameterNames[i].getName()).append(",");
            }
            String s = Modifier.toString(f.getModifiers())+" "+f.getReturnType().getSimpleName()+
                    " "+f.getName()+"("+ (parameterBuilder.length()>0 ? parameterBuilder.substring(0,parameterBuilder.length()-1) : "")+
                    "){some code...}";

            allMethods.add(s.contains("abstract")?s.replace("{some code...}",";"):s);
        }
        for (Method f :
                declaredMethods) {
            Parameter[] parameterNames = f.getParameters();
            Class[] parameterTypes = f.getParameterTypes();
            StringBuilder parameterBuilder = new StringBuilder();
            for (int i=0;i<parameterNames.length;i++){
                parameterBuilder.append(parameterTypes[i].getSimpleName()).append(" ").append(parameterNames[i].getName()).append(",");
            }
            String s = Modifier.toString(f.getModifiers())+" "+f.getReturnType().getSimpleName()+
                    " "+f.getName()+"("+ (parameterBuilder.length()>0 ? parameterBuilder.substring(0,parameterBuilder.length()-1) : "")+
                    "){some code...}";
            if (!allMethods.contains(s))
                allMethods.add(s.contains("abstract")?s.replace("{some code...}",";"):s);
        }
        if (c.getSuperclass() != null) {
            getAllClassMethods(allMethods, c.getSuperclass());
        }
        removeDuplicateFromList(allMethods);
    }
    private static void getAllClassConstructors(List<String> allConstructors, Class c){
        Constructor[] constructors = c.getConstructors();
        Constructor[] declaredConstructors = c.getDeclaredConstructors();
        for (Constructor f :
                constructors) {
            Parameter[] parameterNames = f.getParameters();
            Class[] parameterTypes = f.getParameterTypes();
            StringBuilder parameterBuilder = new StringBuilder();
            for (int i=0;i<parameterNames.length;i++){
                parameterBuilder.append(parameterTypes[i].getSimpleName()).append(" ").append(parameterNames[i].getName()).append(",");
            }
            allConstructors.add(Modifier.toString(f.getModifiers())+" "
                    +f.getName()+"("+ (parameterBuilder.length()>0 ? parameterBuilder.substring(0,parameterBuilder.length()-1) : "")+
                    "){some code...}");
        }
        for (Constructor f :
                declaredConstructors) {
            Parameter[] parameterNames = f.getParameters();
            Class[] parameterTypes = f.getParameterTypes();
            StringBuilder parameterBuilder = new StringBuilder();
            for (int i=0;i<parameterNames.length;i++){
                parameterBuilder.append(parameterTypes[i].getSimpleName()).append(" ").append(parameterNames[i].getName()).append(",");
            }
            String s = Modifier.toString(f.getModifiers())+" "+
                    f.getName()+"("+ (parameterBuilder.length()>0 ? parameterBuilder.substring(0,parameterBuilder.length()-1) : "")+
                    "){some code...}";
            if (!allConstructors.contains(s))
                allConstructors.add(s);
        }
    }
    private static <T> void removeDuplicateFromList(List<T> list){
        Set<T> set = new HashSet<>(list);
        list.clear();
        list.addAll(set);
    }
    public static void main(String[] args) {
        new Form1();
    }
}
