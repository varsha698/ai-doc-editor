import type { GetServerSideProps } from "next";

export default function Home() {
  return null;
}

export const getServerSideProps: GetServerSideProps = async () => ({
  redirect: { destination: "/dashboard", permanent: false },
});
