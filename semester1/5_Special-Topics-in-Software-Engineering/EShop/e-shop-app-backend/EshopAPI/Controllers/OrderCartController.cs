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
    public class OrderCartController : Controller
    {
        private readonly EshopDbv2Context _context;
        private readonly IConfiguration _configuration;

        public OrderCartController(EshopDbv2Context context, IConfiguration configuration) 
        {
            _context = context;
            _configuration = configuration;
        }

        [HttpGet]
        public JsonResult Get()
        {
            List<OrderCart> orderCarts = _context.OrderCarts.ToList();
            foreach (var orderCart in orderCarts)
            {
                CartController cartController = new CartController(_context, _configuration);
                EshopUserController eshopUserController = new EshopUserController(_context, _configuration);
                orderCart.CartNavigation = cartController.GetByCartId(orderCart.Cart);
                orderCart.CustomerNavigation = eshopUserController.GetByEshopUserId(orderCart.Customer);

            }
            return new JsonResult(orderCarts);
        }

        [HttpGet("{id}")]
        public JsonResult GetById(int id)
        {
            OrderCart orderCart = _context.OrderCarts.Single(a => a.Id == id);
            return new JsonResult(orderCart);
        }

        public OrderCart GetByOrderCartId(int id)
        {
            OrderCart orderCart = _context.OrderCarts.Single(a => a.Id == id);
            return orderCart;
        }

        [HttpPost]
        public JsonResult Post(OrderCart orderCart)
        {
            _context.Attach(orderCart);
            _context.Entry(orderCart).State = EntityState.Added;
            _context.SaveChanges();
            return new JsonResult("Inserted Successfully");
        }

        [HttpPut]
        public JsonResult Put(OrderCart orderCart)
        {
            _context.Attach(orderCart);
            _context.Entry(orderCart).State = EntityState.Modified;
            _context.SaveChanges();
            return new JsonResult("Updated Successfully");
        }

        [HttpDelete("{id}")]
        public JsonResult Delete(int id)
        {
            OrderCart orderCart = _context.OrderCarts.Single(a => a.Id == id);
            _context.Attach(orderCart);
            _context.Entry(orderCart).State = EntityState.Deleted;
            _context.SaveChanges();
            return new JsonResult("Deleted Successfully");
        }

    }
}
