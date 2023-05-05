using System;
using System.Collections.Generic;
using EshopAPI.Models;
using Microsoft.EntityFrameworkCore;

namespace EshopAPI.Data;

public partial class EshopDbv2Context : DbContext
{
    public EshopDbv2Context()
    {
    }

    public EshopDbv2Context(DbContextOptions<EshopDbv2Context> options)
        : base(options)
    {
    }

    public virtual DbSet<Admin> Admins { get; set; }

    public virtual DbSet<Cart> Carts { get; set; }

    public virtual DbSet<CartProduct> CartProducts { get; set; }

    public virtual DbSet<EshopUser> EshopUsers { get; set; }

    public virtual DbSet<OrderCart> OrderCarts { get; set; }

    public virtual DbSet<Payment> Payments { get; set; }

    public virtual DbSet<PaymentCategory> PaymentCategories { get; set; }

    public virtual DbSet<Product> Products { get; set; }

    public virtual DbSet<ProductCategory> ProductCategories { get; set; }

    public virtual DbSet<Review> Reviews { get; set; }

    public virtual DbSet<Role> Roles { get; set; }

    public virtual DbSet<Shop> Shops { get; set; }

    public virtual DbSet<ShopCategory> ShopCategories { get; set; }

    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
#warning To protect potentially sensitive information in your connection string, you should move it out of source code. You can avoid scaffolding the connection string by using the Name= syntax to read it from configuration - see https://go.microsoft.com/fwlink/?linkid=2131148. For more guidance on storing connection strings, see http://go.microsoft.com/fwlink/?LinkId=723263.
        => optionsBuilder.UseSqlServer("Server=.; Database=EshopDBv2; User Id=<DESKTOP-7JTCESD\\PX>; Integrated Security=true; Encrypt=false");

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<Admin>(entity =>
        {
            entity.HasKey(e => e.Id).HasName("PK__Admin__3214EC27C6CF9CD1");

            entity.ToTable("Admin");

            entity.Property(e => e.Id).HasColumnName("ID");
            entity.Property(e => e.Description)
                .HasMaxLength(500)
                .IsUnicode(false)
                .HasColumnName("DESCRIPTION");
            entity.Property(e => e.RoleId).HasColumnName("ROLE_ID");
            entity.Property(e => e.UserId).HasColumnName("USER_ID");

            entity.HasOne(d => d.Role).WithMany(p => p.Admins)
                .HasForeignKey(d => d.RoleId)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK__Admin__ROLE_ID__3B75D760");

            entity.HasOne(d => d.User).WithMany(p => p.Admins)
                .HasForeignKey(d => d.UserId)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK__Admin__USER_ID__3C69FB99");
        });

        modelBuilder.Entity<Cart>(entity =>
        {
            entity.HasKey(e => e.Id).HasName("PK__Cart__3214EC2768BFA8FA");

            entity.ToTable("Cart");

            entity.Property(e => e.Id).HasColumnName("ID");
            entity.Property(e => e.Customer).HasColumnName("CUSTOMER");
            entity.Property(e => e.Quantity).HasColumnName("QUANTITY");

            entity.HasOne(d => d.CustomerNavigation).WithMany(p => p.Carts)
                .HasForeignKey(d => d.Customer)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK__Cart__CUSTOMER__4AB81AF0");
        });

        modelBuilder.Entity<CartProduct>(entity =>
        {
            entity.HasKey(e => e.Id).HasName("PK__CartProd__3214EC27058B84A7");

            entity.ToTable("CartProduct");

            entity.Property(e => e.Id).HasColumnName("ID");

            entity.HasOne(d => d.CartNavigation).WithMany(p => p.CartProducts)
                .HasForeignKey(d => d.Cart)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK__CartProduc__Cart__4D94879B");

            entity.HasOne(d => d.ProductNavigation).WithMany(p => p.CartProducts)
                .HasForeignKey(d => d.Product)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK__CartProdu__Produ__4E88ABD4");
        });

        modelBuilder.Entity<EshopUser>(entity =>
        {
            entity.HasKey(e => e.Id).HasName("PK__EshopUse__3214EC27FF563622");

            entity.ToTable("EshopUser");

            entity.Property(e => e.Id).HasColumnName("ID");
            entity.Property(e => e.Address)
                .HasMaxLength(500)
                .IsUnicode(false)
                .HasColumnName("ADDRESS");
            entity.Property(e => e.Email)
                .HasMaxLength(500)
                .IsUnicode(false)
                .HasColumnName("EMAIL");
            entity.Property(e => e.Username)
                .HasMaxLength(500)
                .IsUnicode(false)
                .HasColumnName("USERNAME");
        });

        modelBuilder.Entity<OrderCart>(entity =>
        {
            entity.HasKey(e => e.Id).HasName("PK__OrderCar__3214EC2729649F63");

            entity.ToTable("OrderCart");

            entity.Property(e => e.Id).HasColumnName("ID");
            entity.Property(e => e.Date).HasColumnType("datetime");
            entity.Property(e => e.DeliveryAdress)
                .HasMaxLength(500)
                .IsUnicode(false);

            entity.HasOne(d => d.CartNavigation).WithMany(p => p.OrderCarts)
                .HasForeignKey(d => d.Cart)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK__OrderCart__Cart__5165187F");

            entity.HasOne(d => d.CustomerNavigation).WithMany(p => p.OrderCarts)
                .HasForeignKey(d => d.Customer)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK__OrderCart__Custo__52593CB8");

            entity.HasOne(d => d.PaymentNavigation).WithMany(p => p.OrderCarts)
                .HasForeignKey(d => d.Payment)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK__OrderCart__Payme__534D60F1");
        });

        modelBuilder.Entity<Payment>(entity =>
        {
            entity.HasKey(e => e.Id).HasName("PK__Payment__3214EC2744E869EA");

            entity.ToTable("Payment");

            entity.Property(e => e.Id).HasColumnName("ID");
            entity.Property(e => e.Amount)
                .HasColumnType("decimal(18, 2)")
                .HasColumnName("AMOUNT");
            entity.Property(e => e.Availability).HasColumnName("AVAILABILITY");
            entity.Property(e => e.PaymentCategoryId).HasColumnName("PAYMENT_CATEGORY_ID");

            entity.HasOne(d => d.PaymentCategory).WithMany(p => p.Payments)
                .HasForeignKey(d => d.PaymentCategoryId)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK__Payment__PAYMENT__47DBAE45");
        });

        modelBuilder.Entity<PaymentCategory>(entity =>
        {
            entity.HasKey(e => e.Id).HasName("PK__PaymentC__3214EC27508298FB");

            entity.ToTable("PaymentCategory");

            entity.Property(e => e.Id).HasColumnName("ID");
            entity.Property(e => e.Description)
                .HasMaxLength(500)
                .IsUnicode(false)
                .HasColumnName("DESCRIPTION");
            entity.Property(e => e.Name)
                .HasMaxLength(500)
                .IsUnicode(false)
                .HasColumnName("NAME");
        });

        modelBuilder.Entity<Product>(entity =>
        {
            entity.HasKey(e => e.Id).HasName("PK__Product__3214EC27E1B5F395");

            entity.ToTable("Product");

            entity.Property(e => e.Id).HasColumnName("ID");
            entity.Property(e => e.Availability).HasColumnName("AVAILABILITY");
            entity.Property(e => e.Description)
                .HasMaxLength(500)
                .IsUnicode(false)
                .HasColumnName("DESCRIPTION");
            entity.Property(e => e.Image)
                .HasMaxLength(500)
                .IsUnicode(false)
                .HasColumnName("IMAGE");
            entity.Property(e => e.Name)
                .HasMaxLength(500)
                .IsUnicode(false)
                .HasColumnName("NAME");
            entity.Property(e => e.ProductCategoryId).HasColumnName("PRODUCT_CATEGORY_ID");

            entity.HasOne(d => d.ProductCategory).WithMany(p => p.Products)
                .HasForeignKey(d => d.ProductCategoryId)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK__Product__PRODUCT__44FF419A");
        });

        modelBuilder.Entity<ProductCategory>(entity =>
        {
            entity.HasKey(e => e.Id).HasName("PK__ProductC__3214EC277C910D4F");

            entity.ToTable("ProductCategory");

            entity.Property(e => e.Id).HasColumnName("ID");
            entity.Property(e => e.Description)
                .HasMaxLength(500)
                .IsUnicode(false)
                .HasColumnName("DESCRIPTION");
            entity.Property(e => e.Name)
                .HasMaxLength(500)
                .IsUnicode(false)
                .HasColumnName("NAME");
        });

        modelBuilder.Entity<Review>(entity =>
        {
            entity.HasKey(e => e.Id).HasName("PK__Review__3214EC27B60DBCC7");

            entity.ToTable("Review");

            entity.Property(e => e.Id).HasColumnName("ID");
            entity.Property(e => e.Description)
                .HasMaxLength(500)
                .IsUnicode(false);
            entity.Property(e => e.Rating).HasColumnType("decimal(5, 2)");

            entity.HasOne(d => d.CustomerNavigation).WithMany(p => p.Reviews)
                .HasForeignKey(d => d.Customer)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK__Review__Customer__571DF1D5");

            entity.HasOne(d => d.ProductNavigation).WithMany(p => p.Reviews)
                .HasForeignKey(d => d.Product)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK__Review__Product__5629CD9C");
        });

        modelBuilder.Entity<Role>(entity =>
        {
            entity.HasKey(e => e.Id).HasName("PK__Role__3214EC27ABBEAFFB");

            entity.ToTable("Role");

            entity.Property(e => e.Id).HasColumnName("ID");
            entity.Property(e => e.Description)
                .HasMaxLength(500)
                .IsUnicode(false)
                .HasColumnName("DESCRIPTION");
            entity.Property(e => e.Name)
                .HasMaxLength(500)
                .IsUnicode(false)
                .HasColumnName("NAME");
        });

        modelBuilder.Entity<Shop>(entity =>
        {
            entity.HasKey(e => e.Id).HasName("PK__Shop__3214EC27F7F876CA");

            entity.ToTable("Shop");

            entity.Property(e => e.Id).HasColumnName("ID");
            entity.Property(e => e.Description)
                .HasMaxLength(500)
                .IsUnicode(false);
            entity.Property(e => e.Image)
                .HasMaxLength(500)
                .IsUnicode(false);
            entity.Property(e => e.Name)
                .HasMaxLength(500)
                .IsUnicode(false);

            entity.HasOne(d => d.SellerNavigation).WithMany(p => p.Shops)
                .HasForeignKey(d => d.Seller)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK__Shop__Seller__5AEE82B9");

            entity.HasOne(d => d.ShopCategoryNavigation).WithMany(p => p.Shops)
                .HasForeignKey(d => d.ShopCategory)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK__Shop__ShopCatego__59FA5E80");
        });

        modelBuilder.Entity<ShopCategory>(entity =>
        {
            entity.HasKey(e => e.Id).HasName("PK__ShopCate__3214EC276BB4DC38");

            entity.ToTable("ShopCategory");

            entity.Property(e => e.Id).HasColumnName("ID");
            entity.Property(e => e.Description)
                .HasMaxLength(500)
                .IsUnicode(false)
                .HasColumnName("DESCRIPTION");
            entity.Property(e => e.Name)
                .HasMaxLength(500)
                .IsUnicode(false)
                .HasColumnName("NAME");
        });

        OnModelCreatingPartial(modelBuilder);
    }

    partial void OnModelCreatingPartial(ModelBuilder modelBuilder);
}
