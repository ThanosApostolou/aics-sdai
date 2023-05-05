using System;
using System.Collections.Generic;

namespace EshopAPI.Models;

public partial class PaymentCategory
{
    public int Id { get; set; }

    public string Name { get; set; } = null!;

    public string Description { get; set; } = null!;

    public virtual ICollection<Payment> Payments { get; } = new List<Payment>();
}
