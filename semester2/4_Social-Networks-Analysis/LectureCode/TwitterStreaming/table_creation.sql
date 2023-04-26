delimiter $$

CREATE DATABASE `stream_monitor` /*!40100 DEFAULT CHARACTER SET utf8 COLLATE utf8_unicode_ci */$$


delimiter $$

CREATE TABLE `data` (
  `TEXT` varchar(200) COLLATE utf8_unicode_ci NOT NULL,
  `SCREEN_NAME` varchar(200) COLLATE utf8_unicode_ci NOT NULL,
  `DATE` datetime NOT NULL,
  `SOURCE` varchar(200) COLLATE utf8_unicode_ci NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci$$