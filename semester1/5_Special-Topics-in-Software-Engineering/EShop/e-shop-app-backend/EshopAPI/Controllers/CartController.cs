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
    public class CartController : Controller
    {
        private readonly EshopDbContext _context;
        private readonly IConfiguration _configuration;

        public CartController(EshopDbContext context, IConfiguration configuration) 
        {
            _context = context;
            _configuration = configuration;
        }

        [HttpGet]
        public JsonResult Get()
        {
            List<Cart> carts = _context.Carts.ToList();
            foreach (var cart in carts)
            {
                EshopUserController eshopUserController = new EshopUserController(_context, _configuration);
                cart.CustomerNavigation = eshopUserController.GetByEshopUserId(cart.Customer);
            }
            return new JsonResult(carts);
        }

        [HttpGet("{id}")]
        public JsonResult GetById(int id)
        {
            Cart cart = _context.Carts.Single(a => a.Id == id);
            return new JsonResult(cart);
        }

        public Cart GetByCartId(int id)
        {
            Cart cart = _context.Carts.Single(a => a.Id == id);
            return cart;
        }

        [HttpPost]
        public JsonResult Post(Cart cart)
        {
            _context.Attach(cart);
            _context.Entry(cart).State = EntityState.Added;
            _context.SaveChanges();
            return new JsonResult("Inserted Successfully");
        }

        [HttpPut]
        public JsonResult Put(Cart cart)
        {
            _context.Attach(cart);
            _context.Entry(cart).State = EntityState.Modified;
            _context.SaveChanges();
            return new JsonResult("Updated Successfully");
        }

        [HttpDelete("{id}")]
        public JsonResult Delete(int id)
        {
            Cart cart = _context.Carts.Single(a => a.Id == id);
            _context.Attach(cart);
            _context.Entry(cart).State = EntityState.Deleted;
            _context.SaveChanges();
            return new JsonResult("Deleted Successfully");
        }

    }
}
