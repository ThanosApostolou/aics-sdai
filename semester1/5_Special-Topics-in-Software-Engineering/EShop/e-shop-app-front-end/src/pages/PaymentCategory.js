import PaymentCategoryDataDisplay from '../components/PaymentCategory/PaymentCategoryTable'
import BottomBar from "../commons/BottomBar"
import { Toaster } from "react-hot-toast";

function PaymentCategory() {
  return (
    <div className="container">
      <h2>Our Payment Categories!!!</h2>
      <span><br /><br /></span>
      <Toaster/>
      <PaymentCategoryDataDisplay />
      <div className="nav-bar-container-light">
        <BottomBar />
      </div>
    </div>
  );
}
export default PaymentCategory;