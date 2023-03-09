package org.example.derby;

import java.sql.*;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Main {

    public static void main(String[] args) {
        //createTableAndData();
        selectAll();
        //insertNewUser(3,"Aggeliki","Theodwrou");
    }
    private static void insertNewUser(int id, String username, String password){
        try {
            Connection connection = connect();
            String insertSQL = "INSERT INTO D_USER VALUES(?,?,?)";
            PreparedStatement preparedStatement = connection.prepareStatement(insertSQL);
            preparedStatement.setInt(1, id);
            preparedStatement.setString(2, username);
            preparedStatement.setString(3, password);
            int count = preparedStatement.executeUpdate();
            if(count>0){
                System.out.println(count+" record updated");
            }
            preparedStatement.close();
            connection.close();
            System.out.println("Done!");
        } catch (SQLException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    private static void selectAll(){
        try {
            Connection connection = connect();
            Statement statement = connection.createStatement();
            String selectSQL = "select * from D_USER";
            ResultSet resultSet = statement.executeQuery(selectSQL);
            while(resultSet.next()){
                System.out.println(resultSet.getString("USERNAME")+","+resultSet.getString("PASSWORD"));
            }
            statement.close();
            connection.close();
            System.out.println("Done!");
        } catch (SQLException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    private static void createTableAndData(){
        try {
            Connection connection = connect();
            String createTableSQL = "CREATE TABLE D_USER"
                    + "(ID INTEGER NOT NULL PRIMARY KEY,"
                    + "USERNAME VARCHAR(20),"
                    + "PASSWORD VARCHAR(20))";
            Statement statement = connection.createStatement();
            statement.executeUpdate(createTableSQL);
            String insertSQL = "INSERT INTO D_USER VALUES(2,'PANTELIS','P12345')";
            statement.executeUpdate(insertSQL);
            statement.close();
            connection.close();
            System.out.println("Done!");
        } catch (SQLException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    private static Connection connect(){
        String connectionString = "jdbc:derby:javaprotal5;create=true";
        Connection connection = null;
        try {
            connection = DriverManager.getConnection(connectionString);
        } catch (SQLException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        return connection;
    }
}

