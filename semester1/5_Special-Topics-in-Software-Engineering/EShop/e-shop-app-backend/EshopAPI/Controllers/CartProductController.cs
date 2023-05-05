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
    public class CartProductController : Controller
    {
        private readonly EshopDbv2Context _context;
        private readonly IConfiguration _configuration;

        public CartProductController(EshopDbv2Context context, IConfiguration configuration) 
        {
            _context = context;
            _configuration = configuration;
        }

        [HttpGet]
        public JsonResult Get()
        {
            List<CartProduct> cartProducts = _context.CartProducts.ToList();
            foreach (var cartProduct in cartProducts)
            {
                CartController cartController = new CartController(_context, _configuration);
                ProductController productController = new ProductController(_context, _configuration);
                cartProduct.CartNavigation = cartController.GetByCartId(cartProduct.Cart);
                cartProduct.ProductNavigation = productController.GetByProductId(cartProduct.Product);

            }
            return new JsonResult(cartProducts);
        }

        [HttpGet("{id}")]
        public JsonResult GetById(int id)
        {
            CartProduct cartProduct = _context.CartProducts.Single(a => a.Id == id);
            return new JsonResult(cartProduct);
        }

        public CartProduct GetByCartProductId(int id)
        {
            CartProduct cartProduct = _context.CartProducts.Single(a => a.Id == id);
            return cartProduct;
        }

        [HttpPost]
        public JsonResult Post(CartProduct cartProduct)
        {
            _context.Attach(cartProduct);
            _context.Entry(cartProduct).State = EntityState.Added;
            _context.SaveChanges();
            return new JsonResult("Inserted Successfully");
        }

        [HttpPut]
        public JsonResult Put(CartProduct cartProduct)
        {
            _context.Attach(cartProduct);
            _context.Entry(cartProduct).State = EntityState.Modified;
            _context.SaveChanges();
            return new JsonResult("Updated Successfully");
        }

        [HttpDelete("{id}")]
        public JsonResult Delete(int id)
        {
            CartProduct cartProduct = _context.CartProducts.Single(a => a.Id == id);
            _context.Attach(cartProduct);
            _context.Entry(cartProduct).State = EntityState.Deleted;
            _context.SaveChanges();
            return new JsonResult("Deleted Successfully");
        }

    }
}
