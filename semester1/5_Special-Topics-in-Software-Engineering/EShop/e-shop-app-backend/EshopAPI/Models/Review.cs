using System;
using System.Collections.Generic;

namespace EshopAPI.Models;

public partial class Review
{
    public int Id { get; set; }

    public int Customer { get; set; }

    public int Product { get; set; }

    public decimal Rating { get; set; }

    public string Description { get; set; } = null!;

    public virtual EshopUser CustomerNavigation { get; set; } = null!;

    public virtual Product ProductNavigation { get; set; } = null!;
}
