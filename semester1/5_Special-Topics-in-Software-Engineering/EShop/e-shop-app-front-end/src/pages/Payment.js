import PaymentDataDisplay from '../components/Payment/PaymentTable'
import BottomBar from "../commons/BottomBar"
import { Toaster } from "react-hot-toast";

function Payment() {
  return (
    <div className="container">
      <h2>Our Payments!!!</h2>
      <span><br /><br /></span>
      <Toaster/>
      <PaymentDataDisplay />
      <div className="nav-bar-container-light">
        <BottomBar />
      </div>
    </div>
  );
}
export default Payment;