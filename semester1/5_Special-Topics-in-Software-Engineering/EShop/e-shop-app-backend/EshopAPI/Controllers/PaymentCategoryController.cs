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
    public class ProductCategoryController : Controller
    {
        private readonly EshopDbContext _context;
        private readonly IConfiguration _configuration;

        public ProductCategoryController(EshopDbContext context, IConfiguration configuration) 
        {
            _context = context;
            _configuration = configuration;
        }

        [HttpGet]
        public JsonResult Get()
        {
            List<ProductCategory> productCategories = _context.ProductCategories.ToList();
            return new JsonResult(productCategories);
        }

        [HttpGet("{id}")]
        public JsonResult GetById(int id)
        {
            ProductCategory productCategory = _context.ProductCategories.Single(a => a.Id == id);
            return new JsonResult(productCategory);
        }

        public ProductCategory GetByProductCategoryId(int id)
        {
            ProductCategory productCategory = _context.ProductCategories.Single(a => a.Id == id);
            return productCategory;
        }

        [HttpPost]
        public JsonResult Post(ProductCategory productCategory)
        {
            _context.Attach(productCategory);
            _context.Entry(productCategory).State = EntityState.Added;
            _context.SaveChanges();
            return new JsonResult("Inserted Successfully");
        }

        [HttpPut]
        public JsonResult Put(ProductCategory productCategory)
        {
            _context.Attach(productCategory);
            _context.Entry(productCategory).State = EntityState.Modified;
            _context.SaveChanges();
            return new JsonResult("Updated Successfully");
        }

        [HttpDelete("{id}")]
        public JsonResult Delete(int id)
        {
            ProductCategory productCategory = _context.ProductCategories.Single(a => a.Id == id);
            _context.Attach(productCategory);
            _context.Entry(productCategory).State = EntityState.Deleted;
            _context.SaveChanges();
            return new JsonResult("Deleted Successfully");
        }

    }
}
