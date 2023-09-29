using System;
using Microsoft.EntityFrameworkCore.Metadata;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace EshopAPI.Migrations
{
    /// <inheritdoc />
    public partial class InitialCreate : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AlterDatabase()
                .Annotation("MySql:CharSet", "utf8mb4");

            migrationBuilder.CreateTable(
                name: "EshopUser",
                columns: table => new
                {
                    ID = table.Column<int>(type: "int", nullable: false)
                        .Annotation("MySql:ValueGenerationStrategy", MySqlValueGenerationStrategy.IdentityColumn),
                    USERNAME = table.Column<string>(type: "varchar(500)", unicode: false, maxLength: 500, nullable: false)
                        .Annotation("MySql:CharSet", "utf8mb4"),
                    EMAIL = table.Column<string>(type: "varchar(500)", unicode: false, maxLength: 500, nullable: false)
                        .Annotation("MySql:CharSet", "utf8mb4"),
                    ADDRESS = table.Column<string>(type: "varchar(500)", unicode: false, maxLength: 500, nullable: false)
                        .Annotation("MySql:CharSet", "utf8mb4")
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK__EshopUse__3214EC27FF563622", x => x.ID);
                })
                .Annotation("MySql:CharSet", "utf8mb4");

            migrationBuilder.CreateTable(
                name: "PaymentCategory",
                columns: table => new
                {
                    ID = table.Column<int>(type: "int", nullable: false)
                        .Annotation("MySql:ValueGenerationStrategy", MySqlValueGenerationStrategy.IdentityColumn),
                    NAME = table.Column<string>(type: "varchar(500)", unicode: false, maxLength: 500, nullable: false)
                        .Annotation("MySql:CharSet", "utf8mb4"),
                    DESCRIPTION = table.Column<string>(type: "varchar(500)", unicode: false, maxLength: 500, nullable: false)
                        .Annotation("MySql:CharSet", "utf8mb4")
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK__PaymentC__3214EC27508298FB", x => x.ID);
                })
                .Annotation("MySql:CharSet", "utf8mb4");

            migrationBuilder.CreateTable(
                name: "ProductCategory",
                columns: table => new
                {
                    ID = table.Column<int>(type: "int", nullable: false)
                        .Annotation("MySql:ValueGenerationStrategy", MySqlValueGenerationStrategy.IdentityColumn),
                    NAME = table.Column<string>(type: "varchar(500)", unicode: false, maxLength: 500, nullable: false)
                        .Annotation("MySql:CharSet", "utf8mb4"),
                    DESCRIPTION = table.Column<string>(type: "varchar(500)", unicode: false, maxLength: 500, nullable: false)
                        .Annotation("MySql:CharSet", "utf8mb4")
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK__ProductC__3214EC277C910D4F", x => x.ID);
                })
                .Annotation("MySql:CharSet", "utf8mb4");

            migrationBuilder.CreateTable(
                name: "Role",
                columns: table => new
                {
                    ID = table.Column<int>(type: "int", nullable: false)
                        .Annotation("MySql:ValueGenerationStrategy", MySqlValueGenerationStrategy.IdentityColumn),
                    NAME = table.Column<string>(type: "varchar(500)", unicode: false, maxLength: 500, nullable: false)
                        .Annotation("MySql:CharSet", "utf8mb4"),
                    DESCRIPTION = table.Column<string>(type: "varchar(500)", unicode: false, maxLength: 500, nullable: false)
                        .Annotation("MySql:CharSet", "utf8mb4")
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK__Role__3214EC27ABBEAFFB", x => x.ID);
                })
                .Annotation("MySql:CharSet", "utf8mb4");

            migrationBuilder.CreateTable(
                name: "ShopCategory",
                columns: table => new
                {
                    ID = table.Column<int>(type: "int", nullable: false)
                        .Annotation("MySql:ValueGenerationStrategy", MySqlValueGenerationStrategy.IdentityColumn),
                    NAME = table.Column<string>(type: "varchar(500)", unicode: false, maxLength: 500, nullable: false)
                        .Annotation("MySql:CharSet", "utf8mb4"),
                    DESCRIPTION = table.Column<string>(type: "varchar(500)", unicode: false, maxLength: 500, nullable: false)
                        .Annotation("MySql:CharSet", "utf8mb4")
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK__ShopCate__3214EC276BB4DC38", x => x.ID);
                })
                .Annotation("MySql:CharSet", "utf8mb4");

            migrationBuilder.CreateTable(
                name: "Cart",
                columns: table => new
                {
                    ID = table.Column<int>(type: "int", nullable: false)
                        .Annotation("MySql:ValueGenerationStrategy", MySqlValueGenerationStrategy.IdentityColumn),
                    QUANTITY = table.Column<int>(type: "int", nullable: false),
                    CUSTOMER = table.Column<int>(type: "int", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK__Cart__3214EC2768BFA8FA", x => x.ID);
                    table.ForeignKey(
                        name: "FK__Cart__CUSTOMER__4AB81AF0",
                        column: x => x.CUSTOMER,
                        principalTable: "EshopUser",
                        principalColumn: "ID");
                })
                .Annotation("MySql:CharSet", "utf8mb4");

            migrationBuilder.CreateTable(
                name: "Payment",
                columns: table => new
                {
                    ID = table.Column<int>(type: "int", nullable: false)
                        .Annotation("MySql:ValueGenerationStrategy", MySqlValueGenerationStrategy.IdentityColumn),
                    AVAILABILITY = table.Column<bool>(type: "tinyint(1)", nullable: false),
                    AMOUNT = table.Column<decimal>(type: "decimal(18,2)", nullable: false),
                    PAYMENTCATEGORYID = table.Column<int>(name: "PAYMENT_CATEGORY_ID", type: "int", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK__Payment__3214EC2744E869EA", x => x.ID);
                    table.ForeignKey(
                        name: "FK__Payment__PAYMENT__47DBAE45",
                        column: x => x.PAYMENTCATEGORYID,
                        principalTable: "PaymentCategory",
                        principalColumn: "ID");
                })
                .Annotation("MySql:CharSet", "utf8mb4");

            migrationBuilder.CreateTable(
                name: "Product",
                columns: table => new
                {
                    ID = table.Column<int>(type: "int", nullable: false)
                        .Annotation("MySql:ValueGenerationStrategy", MySqlValueGenerationStrategy.IdentityColumn),
                    NAME = table.Column<string>(type: "varchar(500)", unicode: false, maxLength: 500, nullable: false)
                        .Annotation("MySql:CharSet", "utf8mb4"),
                    DESCRIPTION = table.Column<string>(type: "varchar(500)", unicode: false, maxLength: 500, nullable: false)
                        .Annotation("MySql:CharSet", "utf8mb4"),
                    IMAGE = table.Column<string>(type: "varchar(500)", unicode: false, maxLength: 500, nullable: false)
                        .Annotation("MySql:CharSet", "utf8mb4"),
                    AVAILABILITY = table.Column<bool>(type: "tinyint(1)", nullable: false),
                    PRODUCTCATEGORYID = table.Column<int>(name: "PRODUCT_CATEGORY_ID", type: "int", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK__Product__3214EC27E1B5F395", x => x.ID);
                    table.ForeignKey(
                        name: "FK__Product__PRODUCT__44FF419A",
                        column: x => x.PRODUCTCATEGORYID,
                        principalTable: "ProductCategory",
                        principalColumn: "ID");
                })
                .Annotation("MySql:CharSet", "utf8mb4");

            migrationBuilder.CreateTable(
                name: "Admin",
                columns: table => new
                {
                    ID = table.Column<int>(type: "int", nullable: false)
                        .Annotation("MySql:ValueGenerationStrategy", MySqlValueGenerationStrategy.IdentityColumn),
                    ROLEID = table.Column<int>(name: "ROLE_ID", type: "int", nullable: false),
                    USERID = table.Column<int>(name: "USER_ID", type: "int", nullable: false),
                    DESCRIPTION = table.Column<string>(type: "varchar(500)", unicode: false, maxLength: 500, nullable: false)
                        .Annotation("MySql:CharSet", "utf8mb4")
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK__Admin__3214EC27C6CF9CD1", x => x.ID);
                    table.ForeignKey(
                        name: "FK__Admin__ROLE_ID__3B75D760",
                        column: x => x.ROLEID,
                        principalTable: "Role",
                        principalColumn: "ID");
                    table.ForeignKey(
                        name: "FK__Admin__USER_ID__3C69FB99",
                        column: x => x.USERID,
                        principalTable: "EshopUser",
                        principalColumn: "ID");
                })
                .Annotation("MySql:CharSet", "utf8mb4");

            migrationBuilder.CreateTable(
                name: "Shop",
                columns: table => new
                {
                    ID = table.Column<int>(type: "int", nullable: false)
                        .Annotation("MySql:ValueGenerationStrategy", MySqlValueGenerationStrategy.IdentityColumn),
                    Seller = table.Column<int>(type: "int", nullable: false),
                    ShopCategory = table.Column<int>(type: "int", nullable: false),
                    Name = table.Column<string>(type: "varchar(500)", unicode: false, maxLength: 500, nullable: false)
                        .Annotation("MySql:CharSet", "utf8mb4"),
                    Image = table.Column<string>(type: "varchar(500)", unicode: false, maxLength: 500, nullable: false)
                        .Annotation("MySql:CharSet", "utf8mb4"),
                    Description = table.Column<string>(type: "varchar(500)", unicode: false, maxLength: 500, nullable: false)
                        .Annotation("MySql:CharSet", "utf8mb4")
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK__Shop__3214EC27F7F876CA", x => x.ID);
                    table.ForeignKey(
                        name: "FK__Shop__Seller__5AEE82B9",
                        column: x => x.Seller,
                        principalTable: "EshopUser",
                        principalColumn: "ID");
                    table.ForeignKey(
                        name: "FK__Shop__ShopCatego__59FA5E80",
                        column: x => x.ShopCategory,
                        principalTable: "ShopCategory",
                        principalColumn: "ID");
                })
                .Annotation("MySql:CharSet", "utf8mb4");

            migrationBuilder.CreateTable(
                name: "OrderCart",
                columns: table => new
                {
                    ID = table.Column<int>(type: "int", nullable: false)
                        .Annotation("MySql:ValueGenerationStrategy", MySqlValueGenerationStrategy.IdentityColumn),
                    Customer = table.Column<int>(type: "int", nullable: false),
                    Cart = table.Column<int>(type: "int", nullable: false),
                    Payment = table.Column<int>(type: "int", nullable: false),
                    Date = table.Column<DateTime>(type: "datetime", nullable: false),
                    DeliveryAdress = table.Column<string>(type: "varchar(500)", unicode: false, maxLength: 500, nullable: false)
                        .Annotation("MySql:CharSet", "utf8mb4")
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK__OrderCar__3214EC2729649F63", x => x.ID);
                    table.ForeignKey(
                        name: "FK__OrderCart__Cart__5165187F",
                        column: x => x.Cart,
                        principalTable: "Cart",
                        principalColumn: "ID");
                    table.ForeignKey(
                        name: "FK__OrderCart__Custo__52593CB8",
                        column: x => x.Customer,
                        principalTable: "EshopUser",
                        principalColumn: "ID");
                    table.ForeignKey(
                        name: "FK__OrderCart__Payme__534D60F1",
                        column: x => x.Payment,
                        principalTable: "Payment",
                        principalColumn: "ID");
                })
                .Annotation("MySql:CharSet", "utf8mb4");

            migrationBuilder.CreateTable(
                name: "CartProduct",
                columns: table => new
                {
                    ID = table.Column<int>(type: "int", nullable: false)
                        .Annotation("MySql:ValueGenerationStrategy", MySqlValueGenerationStrategy.IdentityColumn),
                    Cart = table.Column<int>(type: "int", nullable: false),
                    Product = table.Column<int>(type: "int", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK__CartProd__3214EC27058B84A7", x => x.ID);
                    table.ForeignKey(
                        name: "FK__CartProdu__Produ__4E88ABD4",
                        column: x => x.Product,
                        principalTable: "Product",
                        principalColumn: "ID");
                    table.ForeignKey(
                        name: "FK__CartProduc__Cart__4D94879B",
                        column: x => x.Cart,
                        principalTable: "Cart",
                        principalColumn: "ID");
                })
                .Annotation("MySql:CharSet", "utf8mb4");

            migrationBuilder.CreateTable(
                name: "Review",
                columns: table => new
                {
                    ID = table.Column<int>(type: "int", nullable: false)
                        .Annotation("MySql:ValueGenerationStrategy", MySqlValueGenerationStrategy.IdentityColumn),
                    Customer = table.Column<int>(type: "int", nullable: false),
                    Product = table.Column<int>(type: "int", nullable: false),
                    Rating = table.Column<decimal>(type: "decimal(5,2)", nullable: false),
                    Description = table.Column<string>(type: "varchar(500)", unicode: false, maxLength: 500, nullable: false)
                        .Annotation("MySql:CharSet", "utf8mb4")
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK__Review__3214EC27B60DBCC7", x => x.ID);
                    table.ForeignKey(
                        name: "FK__Review__Customer__571DF1D5",
                        column: x => x.Customer,
                        principalTable: "EshopUser",
                        principalColumn: "ID");
                    table.ForeignKey(
                        name: "FK__Review__Product__5629CD9C",
                        column: x => x.Product,
                        principalTable: "Product",
                        principalColumn: "ID");
                })
                .Annotation("MySql:CharSet", "utf8mb4");

            migrationBuilder.CreateIndex(
                name: "IX_Admin_ROLE_ID",
                table: "Admin",
                column: "ROLE_ID");

            migrationBuilder.CreateIndex(
                name: "IX_Admin_USER_ID",
                table: "Admin",
                column: "USER_ID");

            migrationBuilder.CreateIndex(
                name: "IX_Cart_CUSTOMER",
                table: "Cart",
                column: "CUSTOMER");

            migrationBuilder.CreateIndex(
                name: "IX_CartProduct_Cart",
                table: "CartProduct",
                column: "Cart");

            migrationBuilder.CreateIndex(
                name: "IX_CartProduct_Product",
                table: "CartProduct",
                column: "Product");

            migrationBuilder.CreateIndex(
                name: "IX_OrderCart_Cart",
                table: "OrderCart",
                column: "Cart");

            migrationBuilder.CreateIndex(
                name: "IX_OrderCart_Customer",
                table: "OrderCart",
                column: "Customer");

            migrationBuilder.CreateIndex(
                name: "IX_OrderCart_Payment",
                table: "OrderCart",
                column: "Payment");

            migrationBuilder.CreateIndex(
                name: "IX_Payment_PAYMENT_CATEGORY_ID",
                table: "Payment",
                column: "PAYMENT_CATEGORY_ID");

            migrationBuilder.CreateIndex(
                name: "IX_Product_PRODUCT_CATEGORY_ID",
                table: "Product",
                column: "PRODUCT_CATEGORY_ID");

            migrationBuilder.CreateIndex(
                name: "IX_Review_Customer",
                table: "Review",
                column: "Customer");

            migrationBuilder.CreateIndex(
                name: "IX_Review_Product",
                table: "Review",
                column: "Product");

            migrationBuilder.CreateIndex(
                name: "IX_Shop_Seller",
                table: "Shop",
                column: "Seller");

            migrationBuilder.CreateIndex(
                name: "IX_Shop_ShopCategory",
                table: "Shop",
                column: "ShopCategory");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "Admin");

            migrationBuilder.DropTable(
                name: "CartProduct");

            migrationBuilder.DropTable(
                name: "OrderCart");

            migrationBuilder.DropTable(
                name: "Review");

            migrationBuilder.DropTable(
                name: "Shop");

            migrationBuilder.DropTable(
                name: "Role");

            migrationBuilder.DropTable(
                name: "Cart");

            migrationBuilder.DropTable(
                name: "Payment");

            migrationBuilder.DropTable(
                name: "Product");

            migrationBuilder.DropTable(
                name: "ShopCategory");

            migrationBuilder.DropTable(
                name: "EshopUser");

            migrationBuilder.DropTable(
                name: "PaymentCategory");

            migrationBuilder.DropTable(
                name: "ProductCategory");
        }
    }
}
