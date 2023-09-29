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
    public class PaymentCategoryController : Controller
    {
        private readonly EshopDbContext _context;
        private readonly IConfiguration _configuration;

        public PaymentCategoryController(EshopDbContext context, IConfiguration configuration) 
        {
            _context = context;
            _configuration = configuration;
        }

        [HttpGet]
        public JsonResult Get()
        {
            List<PaymentCategory> paymentCategories = _context.PaymentCategories.ToList();
            return new JsonResult(paymentCategories);
        }

        [HttpGet("{id}")]
        public JsonResult GetById(int id)
        {
            PaymentCategory paymentCategory = _context.PaymentCategories.Single(a => a.Id == id);
            return new JsonResult(paymentCategory);
        }

        public PaymentCategory GetByPaymentCategoryId(int id)
        {
            PaymentCategory paymentCategory = _context.PaymentCategories.Single(a => a.Id == id);
            return paymentCategory;
        }

        [HttpPost]
        public JsonResult Post(PaymentCategory paymentCategory)
        {
            _context.Attach(paymentCategory);
            _context.Entry(paymentCategory).State = EntityState.Added;
            _context.SaveChanges();
            return new JsonResult("Inserted Successfully");
        }

        [HttpPut]
        public JsonResult Put(PaymentCategory paymentCategory)
        {
            _context.Attach(paymentCategory);
            _context.Entry(paymentCategory).State = EntityState.Modified;
            _context.SaveChanges();
            return new JsonResult("Updated Successfully");
        }

        [HttpDelete("{id}")]
        public JsonResult Delete(int id)
        {
            PaymentCategory paymentCategory = _context.PaymentCategories.Single(a => a.Id == id);
            _context.Attach(paymentCategory);
            _context.Entry(paymentCategory).State = EntityState.Deleted;
            _context.SaveChanges();
            return new JsonResult("Deleted Successfully");
        }

    }
}
