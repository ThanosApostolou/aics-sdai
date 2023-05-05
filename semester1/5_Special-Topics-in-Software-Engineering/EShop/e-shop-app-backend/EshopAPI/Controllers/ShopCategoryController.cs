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
    public class ShopCategoryController : Controller
    {
        private readonly EshopDbv2Context _context;
        private readonly IConfiguration _configuration;

        public ShopCategoryController(EshopDbv2Context context, IConfiguration configuration) 
        {
            _context = context;
            _configuration = configuration;
        }

        [HttpGet]
        public JsonResult Get()
        {
            List<ShopCategory> shopCategories = _context.ShopCategories.ToList();
            return new JsonResult(shopCategories);
        }

        [HttpGet("{id}")]
        public JsonResult GetById(int id)
        {
            ShopCategory shopCategory = _context.ShopCategories.Single(a => a.Id == id);
            return new JsonResult(shopCategory);
        }

        public ShopCategory GetByShopCategoryId(int id)
        {
            ShopCategory shopCategory = _context.ShopCategories.Single(a => a.Id == id);
            return shopCategory;
        }

        [HttpPost]
        public JsonResult Post(ShopCategory shopCategory)
        {
            _context.Attach(shopCategory);
            _context.Entry(shopCategory).State = EntityState.Added;
            _context.SaveChanges();
            return new JsonResult("Inserted Successfully");
        }

        [HttpPut]
        public JsonResult Put(ShopCategory shopCategory)
        {
            _context.Attach(shopCategory);
            _context.Entry(shopCategory).State = EntityState.Modified;
            _context.SaveChanges();
            return new JsonResult("Updated Successfully");
        }

        [HttpDelete("{id}")]
        public JsonResult Delete(int id)
        {
            ShopCategory shopCategory = _context.ShopCategories.Single(a => a.Id == id);
            _context.Attach(shopCategory);
            _context.Entry(shopCategory).State = EntityState.Deleted;
            _context.SaveChanges();
            return new JsonResult("Deleted Successfully");
        }

    }
}
