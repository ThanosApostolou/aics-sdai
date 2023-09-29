package aics.server.not_found;

import jakarta.ws.rs.NotFoundException;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.ext.ExceptionMapper;
import jakarta.ws.rs.ext.Provider;
import org.jboss.resteasy.reactive.RestResponse;

import java.nio.charset.StandardCharsets;
import java.util.Objects;
import java.util.Scanner;

@Provider
public class NotFoundExceptionMapper implements ExceptionMapper<NotFoundException> {
    @Override
    @Produces(MediaType.TEXT_HTML)
    public Response toResponse(NotFoundException exception) {
        String text = new Scanner(Objects.requireNonNull(this.getClass().getResourceAsStream("/META-INF/resources/index.html")), StandardCharsets.UTF_8).useDelimiter("\\A").next();
        if (text == null) {
            return Response.status(RestResponse.Status.NOT_FOUND).build();
        } else {
            return Response.ok().header("Content-Type", "text/html").entity(text).build();
        }
    }
}