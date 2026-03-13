/*
 Navicat Premium Data Transfer

 Source Server         : mysql
 Source Server Type    : MySQL
 Source Server Version : 50538
 Source Host           : localhost:3306
 Source Schema         : rag_agent

 Target Server Type    : MySQL
 Target Server Version : 50538
 File Encoding         : 65001

 Date: 13/03/2026 11:47:26
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for department
-- ----------------------------
DROP TABLE IF EXISTS `department`;
CREATE TABLE `department`  (
  `dept_id` int(11) NOT NULL AUTO_INCREMENT COMMENT '部门id',
  `dept_name` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '部门名称',
  PRIMARY KEY (`dept_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 4 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Compact;

-- ----------------------------
-- Records of department
-- ----------------------------
INSERT INTO `department` VALUES (1, 'OE');
INSERT INTO `department` VALUES (2, 'TQ');
INSERT INTO `department` VALUES (3, 'HR');

-- ----------------------------
-- Table structure for file
-- ----------------------------
DROP TABLE IF EXISTS `file`;
CREATE TABLE `file`  (
  `file_id` int(11) NOT NULL AUTO_INCREMENT COMMENT '文件id',
  `user_id` int(11) NULL DEFAULT NULL COMMENT '用户id',
  `dept_id` int(11) NULL DEFAULT NULL COMMENT '部门id',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `file_name` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '文件名',
  `file_path` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '文件路径',
  `file_type` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '文件类型',
  `state` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '状态(0表示被删除，1表示存在)',
  PRIMARY KEY (`file_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 10 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Compact;

-- ----------------------------
-- Records of file
-- ----------------------------
INSERT INTO `file` VALUES (1, 1, 1, '2026-03-12 15:13:38', 'Qwen__20260310_pxgmbzf0q.txt', '/uploads/OE/Qwen__20260310_pxgmbzf0q.txt', 'txt', '1');
INSERT INTO `file` VALUES (2, 1, 1, '2026-03-12 17:55:12', 'server_2026-03-12 17_55_12.py', '/uploads/OE/server_2026-03-12 17_55_12.py', 'py', '0');
INSERT INTO `file` VALUES (9, 1, 1, '2026-03-12 18:38:36', 'server.py', '/uploads/OE/server.py', 'py', '1');

-- ----------------------------
-- Table structure for role
-- ----------------------------
DROP TABLE IF EXISTS `role`;
CREATE TABLE `role`  (
  `role_id` int(11) NOT NULL AUTO_INCREMENT COMMENT '权限id',
  `role_name` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '权限名',
  PRIMARY KEY (`role_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 4 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Compact;

-- ----------------------------
-- Records of role
-- ----------------------------
INSERT INTO `role` VALUES (1, 'CEO');
INSERT INTO `role` VALUES (2, 'Supervisor');
INSERT INTO `role` VALUES (3, 'Manager');

-- ----------------------------
-- Table structure for role_department
-- ----------------------------
DROP TABLE IF EXISTS `role_department`;
CREATE TABLE `role_department`  (
  `r_d_id` int(11) NOT NULL AUTO_INCREMENT COMMENT '权限和部门管理的id',
  `role_id` int(11) NULL DEFAULT NULL COMMENT '权限id',
  `dept_id` int(11) NULL DEFAULT NULL COMMENT '部门id',
  PRIMARY KEY (`r_d_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 7 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Compact;

-- ----------------------------
-- Records of role_department
-- ----------------------------
INSERT INTO `role_department` VALUES (1, 1, 1);
INSERT INTO `role_department` VALUES (2, 1, 2);
INSERT INTO `role_department` VALUES (3, 1, 3);
INSERT INTO `role_department` VALUES (4, 2, 1);
INSERT INTO `role_department` VALUES (5, 3, 1);
INSERT INTO `role_department` VALUES (6, 3, 2);

-- ----------------------------
-- Table structure for users
-- ----------------------------
DROP TABLE IF EXISTS `users`;
CREATE TABLE `users`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '用户id',
  `username` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '用户名',
  `password` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '用户密码',
  `dept_id` int(11) NULL DEFAULT NULL COMMENT '部门id',
  `role_id` int(11) NULL DEFAULT NULL COMMENT '权限id',
  `create_time` datetime NULL DEFAULT NULL COMMENT '注册时间',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Compact;

-- ----------------------------
-- Records of users
-- ----------------------------
INSERT INTO `users` VALUES (1, 'EdenXie', 'e10adc3949ba59abbe56e057f20f883e', 1, 1, '2026-03-12 14:27:36');

SET FOREIGN_KEY_CHECKS = 1;
