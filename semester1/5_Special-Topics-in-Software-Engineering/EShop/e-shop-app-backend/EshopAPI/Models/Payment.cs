using System;
using System.Collections.Generic;

namespace EshopAPI.Models;

public partial class Payment
{
    public int Id { get; set; }

    public bool Availability { get; set; }

    public decimal Amount { get; set; }

    public int PaymentCategoryId { get; set; }

    public virtual ICollection<OrderCart> OrderCarts { get; } = new List<OrderCart>();

    public virtual PaymentCategory PaymentCategory { get; set; } = null!;
}
