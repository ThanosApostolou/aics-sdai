using System;
using System.Collections.Generic;

namespace EshopAPI.Models;

public partial class ShopCategory
{
    public int Id { get; set; }

    public string Name { get; set; } = null!;

    public string Description { get; set; } = null!;

    public virtual ICollection<Shop> Shops { get; } = new List<Shop>();
}
