CREATE DATABASE IF NOT EXISTS EshopDB;
USE EshopDB;
CREATE USER IF NOT EXISTS 'EshopDBUser'@'%' identified by 'EshopDBPassword';
GRANT ALL ON EshopDB.* to 'EshopDBUser'@'%';