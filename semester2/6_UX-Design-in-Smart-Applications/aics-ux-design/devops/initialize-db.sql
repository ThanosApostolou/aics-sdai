CREATE DATABASE IF NOT EXISTS newsrecommenderdb;
USE newsrecommenderdb;
CREATE USER IF NOT EXISTS 'newsrecommenderdbUser'@'%' identified by 'newsrecommenderdbPassword';
GRANT ALL ON newsrecommenderdb.* to 'newsrecommenderdbUser'@'%';