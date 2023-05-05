using System;
using System.Collections.Generic;

namespace EshopAPI.Models;

public partial class Admin
{
    public int Id { get; set; }

    public int RoleId { get; set; }

    public int UserId { get; set; }

    public string Description { get; set; } = null!;

    public virtual Role Role { get; set; } = null!;

    public virtual EshopUser User { get; set; } = null!;
}
