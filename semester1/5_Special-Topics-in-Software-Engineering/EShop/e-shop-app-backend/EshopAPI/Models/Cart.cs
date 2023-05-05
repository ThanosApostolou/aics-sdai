using System;
using System.Collections.Generic;

namespace EshopAPI.Models;

public partial class Cart
{
    public int Id { get; set; }

    public int Quantity { get; set; }

    public int Customer { get; set; }

    public virtual ICollection<CartProduct> CartProducts { get; } = new List<CartProduct>();

    public virtual EshopUser CustomerNavigation { get; set; } = null!;

    public virtual ICollection<OrderCart> OrderCarts { get; } = new List<OrderCart>();
}
