using System;
using System.Collections.Generic;

namespace EshopAPI.Models;

public partial class Shop
{
    public int Id { get; set; }

    public int Seller { get; set; }

    public int ShopCategory { get; set; }

    public string Name { get; set; } = null!;

    public string Image { get; set; } = null!;

    public string Description { get; set; } = null!;

    public virtual EshopUser SellerNavigation { get; set; } = null!;

    public virtual ShopCategory ShopCategoryNavigation { get; set; } = null!;
}
