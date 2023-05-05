import BottomBar from "../commons/BottomBar";
import BannerImage from "../images/banner.jpg";
import { Button } from "@mui/material";
import { Link } from "react-router-dom";

function Home() {
  return (
    <div>
      <div
        style={{
          background: `linear-gradient(
            rgba(0, 0, 0, 0.7), 
            rgba(0, 0, 0, 0.7)
          ), url(${BannerImage}) center center/cover no-repeat fixed`,
          height: "80vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div style={{ textAlign: "center", color: "#fff" }}>
          <h1>Welcome to E-Shop!</h1>
          <p>Find everything you need for your daily life!</p>
          <Button
            variant="contained"
            component={Link}
            to="/shop"
            style={{ backgroundColor: "#fff", color: "black" }}
          >
            Start Shopping
          </Button>
        </div>
      </div>
      <div style={{ marginTop: "5rem" }}>
        <h2 style={{ textAlign: "center" }}>Our Mission</h2>
        <p style={{ textAlign: "center" }}>
          Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut euismod,
          eros ac porttitor egestas, nisi ipsum gravida enim, sit amet
          sollicitudin purus nulla vel lorem.
        </p>
      </div>
      <div className="bottom-nav-bar" style={{ marginTop: "5rem" }}>
        <BottomBar />
      </div>
    </div>
  );
}

export default Home;
