import "./globals.css";
import Navbar from "../components/Navbar";

export const metadata = {
  title: "Air Quality Project Dashboard",
  description: "Node/React status dashboard for the air quality repository",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <Navbar />
        {children}
      </body>
    </html>
  );
}
