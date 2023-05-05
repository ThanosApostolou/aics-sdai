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
    public class PaymentController : Controller
    {
        private readonly EshopDbv2Context _context;
        private readonly IConfiguration _configuration;

        public PaymentController(EshopDbv2Context context, IConfiguration configuration) 
        {
            _context = context;
            _configuration = configuration;
        }

        [HttpGet]
        public JsonResult Get()
        {
            List<Payment> payments = _context.Payments.ToList();
            foreach (var payment in payments) {
                PaymentCategoryController paymentCategoryController = new PaymentCategoryController(_context, _configuration);
                payment.PaymentCategory= paymentCategoryController.GetByPaymentCategoryId(payment.PaymentCategoryId);
            }
            return new JsonResult(payments);
        }

        [HttpGet("{id}")]
        public JsonResult GetById(int id)
        {
            Payment payment = _context.Payments.Single(a => a.Id == id);
            return new JsonResult(payment);
        }

        public Payment GetByPaymentId(int id)
        {
            Payment payment = _context.Payments.Single(a => a.Id == id);
            return payment;
        }

        [HttpPost]
        public JsonResult Post(Payment payment)
        {
            _context.Attach(payment);
            _context.Entry(payment).State = EntityState.Added;
            _context.SaveChanges();
            return new JsonResult("Inserted Successfully");
        }

        [HttpPut]
        public JsonResult Put(Payment payment)
        {
            _context.Attach(payment);
            _context.Entry(payment).State = EntityState.Modified;
            _context.SaveChanges();
            return new JsonResult("Updated Successfully");
        }

        [HttpDelete("{id}")]
        public JsonResult Delete(int id)
        {
            Payment payment = _context.Payments.Single(a => a.Id == id);
            _context.Attach(payment);
            _context.Entry(payment).State = EntityState.Deleted;
            _context.SaveChanges();
            return new JsonResult("Deleted Successfully");
        }
    }
}
