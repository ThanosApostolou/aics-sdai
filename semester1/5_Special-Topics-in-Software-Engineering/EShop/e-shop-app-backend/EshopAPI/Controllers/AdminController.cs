using Microsoft.AspNetCore.Mvc;
using EshopAPI.Models;
using EshopAPI.Data;
using Newtonsoft.Json;
using Microsoft.EntityFrameworkCore;
using System.Configuration;

namespace EshopAPI.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class AdminController : Controller
    {
        private readonly EshopDbv2Context _context;
        private readonly IConfiguration _configuration;

        public AdminController(EshopDbv2Context context, IConfiguration configuration) 
        {
            _context = context;
            _configuration = configuration;
        }

        [HttpGet]
        public JsonResult Get()
        {
            List<Admin> admins = _context.Admins.ToList();
            foreach (var admin in admins)
            {
                EshopUserController eshopUserController = new EshopUserController(_context, _configuration);
                RoleController roleController = new RoleController(_context, _configuration);
                admin.User= eshopUserController.GetByEshopUserId(admin.UserId);
                admin.Role = roleController.GetByRoleId(admin.RoleId);
            }
            return new JsonResult(admins);
        }

        [HttpGet("{id}")]
        public JsonResult GetById(int id)
        {
            Admin admin = _context.Admins.Single(a => a.Id == id);
            return new JsonResult(admin);
        }

        public Admin GetByAdminId(int id)
        {
            Admin admin = _context.Admins.Single(a => a.Id == id);
            return admin;
        }

        [HttpPost]
        public JsonResult Post(Admin admin)
        {
            _context.Attach(admin);
            _context.Entry(admin).State = EntityState.Added;
            _context.SaveChanges();
            return new JsonResult("Inserted Successfully");
        }

        [HttpPut]
        public JsonResult Put(Admin admin)
        {
            _context.Attach(admin);
            _context.Entry(admin).State = EntityState.Modified;
            _context.SaveChanges();
            return new JsonResult("Updated Successfully");
        }

        [HttpDelete("{id}")]
        public JsonResult Delete(int id)
        {
            Admin admin = _context.Admins.Single(a => a.Id == id);
            _context.Attach(admin);
            _context.Entry(admin).State = EntityState.Deleted;
            _context.SaveChanges();
            return new JsonResult("Deleted Successfully");
        }

    }
}
