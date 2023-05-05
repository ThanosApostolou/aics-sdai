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
    public class ProductController : Controller
    {
        private readonly EshopDbv2Context _context;
        private readonly IConfiguration _configuration;

        public ProductController(EshopDbv2Context context, IConfiguration configuration) 
        {
            _context = context;
            _configuration = configuration;
        }

        [HttpGet]
        public JsonResult Get()
        {
            List<Product> products= _context.Products.ToList();
            foreach (var product in products) {
                ProductCategoryController productCategoryController = new ProductCategoryController(_context, _configuration);
                product.ProductCategory= productCategoryController.GetByProductCategoryId(product.ProductCategoryId);
            }
            return new JsonResult(products);
        }

        [HttpGet("{id}")]
        public JsonResult GetById(int id)
        {
            Product product = _context.Products.Single(a => a.Id == id);
            return new JsonResult(product);
        }

        public Product GetByProductId(int id)
        {
            Product product = _context.Products.Single(a => a.Id == id);
            return product;
        }

        [HttpPost]
        public JsonResult Post(Product product)
        {
            _context.Attach(product);
            _context.Entry(product).State = EntityState.Added;
            _context.SaveChanges();
            return new JsonResult("Inserted Successfully");
        }

        [HttpPut]
        public JsonResult Put(Product product)
        {
            _context.Attach(product);
            _context.Entry(product).State = EntityState.Modified;
            _context.SaveChanges();
            return new JsonResult("Updated Successfully");
        }

        [HttpDelete("{id}")]
        public JsonResult Delete(int id)
        {
            Product product = _context.Products.Single(a => a.Id == id);
            _context.Attach(product);
            _context.Entry(product).State = EntityState.Deleted;
            _context.SaveChanges();
            return new JsonResult("Deleted Successfully");
        }
    }
}
