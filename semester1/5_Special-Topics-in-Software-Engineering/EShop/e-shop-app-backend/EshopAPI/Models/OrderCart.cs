using System;
using System.Collections.Generic;

namespace EshopAPI.Models;

public partial class OrderCart
{
    public int Id { get; set; }

    public int Customer { get; set; }

    public int Cart { get; set; }

    public int Payment { get; set; }

    public DateTime Date { get; set; }

    public string DeliveryAdress { get; set; } = null!;

    public virtual Cart CartNavigation { get; set; } = null!;

    public virtual EshopUser CustomerNavigation { get; set; } = null!;

    public virtual Payment PaymentNavigation { get; set; } = null!;
}
