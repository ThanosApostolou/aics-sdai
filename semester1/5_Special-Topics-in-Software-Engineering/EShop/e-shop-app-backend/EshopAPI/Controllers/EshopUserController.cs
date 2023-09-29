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
    public class EshopUserController : Controller
    {
        private readonly EshopDbContext _context;
        private readonly IConfiguration _configuration;

        public EshopUserController(EshopDbContext context, IConfiguration configuration) 
        {
            _context = context;
            _configuration = configuration;
        }

        [HttpGet]
        public JsonResult Get()
        {
            List<EshopUser> eshopUsers = _context.EshopUsers.ToList();
            return new JsonResult(eshopUsers);
        }

        [HttpGet("{id}")]
        public JsonResult GetById(int id)
        {
            EshopUser eshopUser = _context.EshopUsers.Single(a => a.Id == id);
            return new JsonResult(eshopUser);
        }

        public EshopUser GetByEshopUserId(int id)
        {
            EshopUser eshopUser = _context.EshopUsers.Single(a => a.Id == id);
            return eshopUser;
        }

        [HttpPost]
        public JsonResult Post(EshopUser eshopUser)
        {
            _context.Attach(eshopUser);
            _context.Entry(eshopUser).State = EntityState.Added;
            _context.SaveChanges();
            return new JsonResult("Inserted Successfully");
        }

        [HttpPut]
        public JsonResult Put(EshopUser eshopUser)
        {
            _context.Attach(eshopUser);
            _context.Entry(eshopUser).State = EntityState.Modified;
            _context.SaveChanges();
            return new JsonResult("Updated Successfully");
        }

        [HttpDelete("{id}")]
        public JsonResult Delete(int id)
        {
            EshopUser eshopUser = _context.EshopUsers.Single(a => a.Id == id);
            _context.Attach(eshopUser);
            _context.Entry(eshopUser).State = EntityState.Deleted;
            _context.SaveChanges();
            return new JsonResult("Deleted Successfully");
        }

    }
}
