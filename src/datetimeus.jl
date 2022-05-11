
import Base.+, Base.show, Base.isless

+(x::Dates.UTInstant{Microsecond}, d::Microsecond) = Dates.UTInstant(x.periods + d)


function show(io::IO, ::MIME"text/plain", x::Dates.UTInstant{Microsecond})
    date = DateTime(Dates.UTM(round(Int, 1e-3 * x.periods.value)))
    show(io, MIME"text/plain"(), date)
end


function DateTimeMicrosecond(y, m, d, h, mi, s, us)
    rata = us + 1_000_000 * (s + 60mi + 3600h + 86400 * Dates.totaldays(y, m, d))
    Dates.UTInstant{Microsecond}(Microsecond(rata))
end


DateTimeMicrosecond(dt::DateTime) = Dates.UTInstant{Microsecond}(Microsecond(1_000 * dt.instant.periods.value))


isless(t1::Dates.UTInstant{Microsecond}, t2::Dates.UTInstant{Microsecond}) = isless(t1.periods.value, t2.periods.value)
