import './index.css' // css import is automatically injected in exported server components

import ClientApp from './ClientApp.jsx'

export function Root(props: { url: URL }) {
  return (
    <html lang="en">
      <head>
        <meta charSet="UTF-8" />
        <link rel="icon" type="image/svg+xml" href="/vite.svg" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Vite + RSC</title>
      </head>
      <body>
  <ClientApp />
      </body>
    </html>
  )
}


