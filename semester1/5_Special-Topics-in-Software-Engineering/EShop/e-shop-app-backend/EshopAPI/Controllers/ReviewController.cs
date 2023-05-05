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
    public class ReviewController : Controller
    {
        private readonly EshopDbv2Context _context;
        private readonly IConfiguration _configuration;

        public ReviewController(EshopDbv2Context context, IConfiguration configuration) 
        {
            _context = context;
            _configuration = configuration;
        }

        [HttpGet]
        public JsonResult Get()
        {
            List<Review> reviews = _context.Reviews.ToList();
            foreach (var review in reviews)
            {
                EshopUserController eshopUserController = new EshopUserController(_context, _configuration);
                review.CustomerNavigation = eshopUserController.GetByEshopUserId(review.Customer);

                ProductController productController = new ProductController(_context, _configuration);
                review.ProductNavigation = productController.GetByProductId(review.Product);
            }
            return new JsonResult(reviews);
        }

        [HttpGet("{id}")]
        public JsonResult GetById(int id)
        {
            Review review = _context.Reviews.Single(a => a.Id == id);
            return new JsonResult(review);
        }

        public Review GetByReviewId(int id)
        {
            Review review = _context.Reviews.Single(a => a.Id == id);
            return review;
        }

        [HttpPost]
        public JsonResult Post(Review review)
        {
            _context.Attach(review);
            _context.Entry(review).State = EntityState.Added;
            _context.SaveChanges();
            return new JsonResult("Inserted Successfully");
        }

        [HttpPut]
        public JsonResult Put(Review review)
        {
            _context.Attach(review);
            _context.Entry(review).State = EntityState.Modified;
            _context.SaveChanges();
            return new JsonResult("Updated Successfully");
        }

        [HttpDelete("{id}")]
        public JsonResult Delete(int id)
        {
            Review review = _context.Reviews.Single(a => a.Id == id);
            _context.Attach(review);
            _context.Entry(review).State = EntityState.Deleted;
            _context.SaveChanges();
            return new JsonResult("Deleted Successfully");
        }

    }
}
