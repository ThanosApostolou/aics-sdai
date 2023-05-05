using System;
using System.Collections.Generic;

namespace EshopAPI.Models;

public partial class Product
{
    public int Id { get; set; }

    public string Name { get; set; } = null!;

    public string Description { get; set; } = null!;

    public string Image { get; set; } = null!;

    public bool Availability { get; set; }

    public int ProductCategoryId { get; set; }

    public virtual ICollection<CartProduct> CartProducts { get; } = new List<CartProduct>();

    public virtual ProductCategory ProductCategory { get; set; } = null!;

    public virtual ICollection<Review> Reviews { get; } = new List<Review>();
}
