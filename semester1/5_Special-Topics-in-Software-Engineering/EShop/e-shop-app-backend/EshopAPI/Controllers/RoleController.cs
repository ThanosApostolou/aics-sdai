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
    public class RoleController : Controller
    {
        private readonly EshopDbv2Context _context;
        private readonly IConfiguration _configuration;

        public RoleController(EshopDbv2Context context, IConfiguration configuration) 
        {
            _context = context;
            _configuration = configuration;
        }

        [HttpGet]
        public JsonResult Get()
        {
            List<Role> roles= _context.Roles.ToList();
            return new JsonResult(roles);
        }

        [HttpGet("{id}")]
        public JsonResult GetById(int id)
        {
            Role role = _context.Roles.Single(a => a.Id == id);
            return new JsonResult(role);
        }

        public Role GetByRoleId(int id)
        {
            Role role = _context.Roles.Single(a => a.Id == id);
            return role;
        }

        [HttpPost]
        public JsonResult Post(Role role)
        {
            _context.Attach(role);
            _context.Entry(role).State = EntityState.Added;
            _context.SaveChanges();
            return new JsonResult("Inserted Successfully");
        }

        [HttpPut]
        public JsonResult Put(Role role)
        {
            _context.Attach(role);
            _context.Entry(role).State = EntityState.Modified;
            _context.SaveChanges();
            return new JsonResult("Updated Successfully");
        }

        [HttpDelete("{id}")]
        public JsonResult Delete(int id)
        {
            Role role = _context.Roles.Single(a => a.Id == id);
            _context.Attach(role);
            _context.Entry(role).State = EntityState.Deleted;
            _context.SaveChanges();
            return new JsonResult("Deleted Successfully");
        }

    }
}
