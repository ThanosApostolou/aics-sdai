using System;
using System.Collections.Generic;

namespace EshopAPI.Models;

public partial class EshopUser
{
    public int Id { get; set; }

    public string Username { get; set; } = null!;

    public string Email { get; set; } = null!;

    public string Address { get; set; } = null!;

    public virtual ICollection<Admin> Admins { get; } = new List<Admin>();

    public virtual ICollection<Cart> Carts { get; } = new List<Cart>();

    public virtual ICollection<OrderCart> OrderCarts { get; } = new List<OrderCart>();

    public virtual ICollection<Review> Reviews { get; } = new List<Review>();

    public virtual ICollection<Shop> Shops { get; } = new List<Shop>();
}
