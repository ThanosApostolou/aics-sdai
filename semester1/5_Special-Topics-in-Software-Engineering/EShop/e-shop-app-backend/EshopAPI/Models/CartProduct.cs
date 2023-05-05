using System;
using System.Collections.Generic;

namespace EshopAPI.Models;

public partial class CartProduct
{
    public int Id { get; set; }

    public int Cart { get; set; }

    public int Product { get; set; }

    public virtual Cart CartNavigation { get; set; } = null!;

    public virtual Product ProductNavigation { get; set; } = null!;
}
