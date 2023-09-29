using Microsoft.AspNetCore.Mvc;
using EshopAPI.Models;
using EshopAPI.Data;
using Newtonsoft.Json;
using System.Data;
using System.Data.SqlClient;
using Microsoft.EntityFrameworkCore;

namespace EshopAPI.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ShopController : Controller
    {
        private readonly EshopDbContext _context;
        private readonly IConfiguration _configuration;

        public ShopController(EshopDbContext context, IConfiguration configuration) 
        {
            _context = context;
            _configuration = configuration;
        }

        [HttpGet]
        public JsonResult Get()
        {
            List<Shop> shops = _context.Shops.ToList();
            foreach (var shop in shops) {
                ShopCategoryController shopCategoryController = new ShopCategoryController(_context, _configuration);
                EshopUserController eshopUserController = new EshopUserController(_context, _configuration);
                shop.ShopCategoryNavigation = shopCategoryController.GetByShopCategoryId(shop.ShopCategory);
                shop.SellerNavigation = eshopUserController.GetByEshopUserId(shop.Seller);
            }
            return new JsonResult(shops);
        }

        [HttpGet("{id}")]
        public JsonResult GetById(int id)
        {
            Shop shop = _context.Shops.Single(a => a.Id == id);
            return new JsonResult(shop);
        }

        [HttpPost]
        public JsonResult Post(Shop shop)
        {
            _context.Attach(shop);
            _context.Entry(shop).State = EntityState.Added;
            _context.SaveChanges();
            return new JsonResult("Inserted Successfully");
        }

        [HttpPut]
        public JsonResult Put(Shop shop)
        {
            _context.Attach(shop);
            _context.Entry(shop).State = EntityState.Modified;
            _context.SaveChanges();
            return new JsonResult("Updated Successfully");
        }

        [HttpDelete("{id}")]
        public JsonResult Delete(int id)
        {
            Shop shop = _context.Shops.Single(a => a.Id == id);
            _context.Attach(shop);
            _context.Entry(shop).State = EntityState.Deleted;
            _context.SaveChanges();
            return new JsonResult("Deleted Successfully");
        }
    }
}
